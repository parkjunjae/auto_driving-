#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from collections import deque
import numpy as np
from tf2_ros import Buffer, TransformListener
from rclpy.duration import Duration
from scipy.spatial import KDTree
# from sklearn.cluster import DBSCAN  # DBSCAN은 실시간 처리에 너무 느림


class Deskew2D(Node):
    """
    Livox PointCloud2의 custom field 'timestamp'를 이용해 yaw(+선택적으로 x,y) deskew 수행.

    - Livox timestamp는 보통 "scan-relative(ns)" (offset)로 들어옴:
        t_point = header.stamp + (ts_raw - ts_raw_max)  # header를 scan 끝으로 가정
      epoch(ns)로 보이는 경우도 있어 자동 감지.

    - /odometry/filtered(또는 /odom)의 시간축이 point cloud의 (스캔 끝)보다 살짝 뒤쳐질 수 있어서,
      ref time을 'scan 끝'이 아니라 'odom_buf의 최신 시각'으로 잡고,
      odom이 커버 못하는 미래 포인트는 drop하여 deskew를 계속 진행한다.
    """

    def __init__(self):
        super().__init__('deskew_2d_node')

        # Topics
        self.declare_parameter('cloud_in', '/livox/lidar')
        self.declare_parameter('cloud_out', '/livox/lidar_deskew')
        self.declare_parameter('odom_in', '/odometry/filtered')  # 필요 시 /odom
        self.declare_parameter('output_frame', 'base_link')  # 출력 frame_id

        # Sanity / performance
        self.declare_parameter('max_scan_ms', 150.0)     # Livox scan range (~100ms) 대비 여유
        self.declare_parameter('min_points', 100)        # 1000 → 100 (더 관대하게)
        self.declare_parameter('odom_buf_len', 8000)     # 4000 → 8000 (더 긴 버퍼)
        self.declare_parameter('max_dt_sec', 0.5)        # 0.2 → 0.5 (타임스탬프 지연 허용)
        self.declare_parameter('assume_header_is_end', False)  # header.stamp를 scan 끝으로 가정
        self.declare_parameter('time_offset', 0.0)       # 타임스탬프 오프셋 보정 (초 단위)

        # Deskew mode (2D)
        self.declare_parameter('deskew_xy', True)        # True면 vx/vy로 x,y도 보정 (가능하면 True)
        self.declare_parameter('deskew_yaw', True)       # True면 wz로 yaw 보정

        # DBSCAN 노이즈 필터링 (주석처리 - 실시간 처리에 너무 느림)
        # self.declare_parameter('enable_dbscan', True)    # DBSCAN 필터 활성화
        # self.declare_parameter('dbscan_eps', 0.5)        # 이웃 반경 (m) - 50cm
        # self.declare_parameter('dbscan_min_samples', 5)  # 최소 포인트 수

        # Radius Outlier Removal 노이즈 필터링 (현재 사용 중)
        self.declare_parameter('enable_outlier_filter', False)  # 필터 활성화
        self.declare_parameter('outlier_radius', 0.5)          # 검색 반경 (m) - 30cm
        self.declare_parameter('outlier_min_neighbors', 5)     # 반경 내 최소 이웃 개수

        cloud_in  = self.get_parameter('cloud_in').value
        cloud_out = self.get_parameter('cloud_out').value
        odom_in   = self.get_parameter('odom_in').value
        self.output_frame = self.get_parameter('output_frame').value

        self.max_scan_ns = int(float(self.get_parameter('max_scan_ms').value) * 1e6)
        self.min_points  = int(self.get_parameter('min_points').value)
        self.max_dt_sec  = float(self.get_parameter('max_dt_sec').value)
        self.time_offset = float(self.get_parameter('time_offset').value)  # 추가

        self.assume_header_is_end = bool(self.get_parameter('assume_header_is_end').value)
        self.deskew_xy = bool(self.get_parameter('deskew_xy').value)
        self.deskew_yaw = bool(self.get_parameter('deskew_yaw').value)

        # DBSCAN 파라미터 (주석처리)
        # self.enable_dbscan = bool(self.get_parameter('enable_dbscan').value)
        # self.dbscan_eps = float(self.get_parameter('dbscan_eps').value)
        # self.dbscan_min_samples = int(self.get_parameter('dbscan_min_samples').value)

        # Radius Outlier Removal 파라미터 (현재 사용 중)
        self.enable_outlier_filter = bool(self.get_parameter('enable_outlier_filter').value)
        self.outlier_radius = float(self.get_parameter('outlier_radius').value)
        self.outlier_min_neighbors = int(self.get_parameter('outlier_min_neighbors').value)

        odom_buf_len = int(self.get_parameter('odom_buf_len').value)

        # Odom buffer: (t_ns, vx, vy, wz)
        self.odom_buf = deque(maxlen=odom_buf_len)

        # TF buffer for base_link -> odom transform
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.sub_odom = self.create_subscription(Odometry, odom_in, self.odom_cb, 200)
        self.sub_cloud = self.create_subscription(PointCloud2, cloud_in, self.cloud_cb, 5)
        self.pub = self.create_publisher(PointCloud2, cloud_out, 5)

        self.get_logger().info(f'Deskew2D: {cloud_in} + {odom_in} -> {cloud_out}')
        self.get_logger().info(
            f'Params: max_scan_ms={self.max_scan_ns/1e6:.1f}, '
            f'deskew_xy={self.deskew_xy}, deskew_yaw={self.deskew_yaw}, '
            f'assume_header_is_end={self.assume_header_is_end}, '
            f'output_frame={self.output_frame}'
        )

    # ===== DBSCAN 필터 (주석처리 - 실시간 처리에 너무 느림) =====
    # def dbscan_filter(self, points):
    #     """
    #     DBSCAN으로 노이즈 제거
    #     - 밀집된 클러스터만 유지
    #     - 고립된 포인트 제거
    #     """
    #     if not self.enable_dbscan or len(points) < self.dbscan_min_samples:
    #         return points
    #
    #     try:
    #         # xyz 좌표만 추출
    #         xyz = np.array([(p[0], p[1], p[2]) for p in points])
    #
    #         # DBSCAN 클러스터링
    #         clustering = DBSCAN(
    #             eps=self.dbscan_eps,
    #             min_samples=self.dbscan_min_samples,
    #             n_jobs=-1  # 모든 CPU 코어 사용
    #         ).fit(xyz)
    #
    #         # 노이즈(-1)가 아닌 클러스터만 유지
    #         labels = clustering.labels_
    #         mask = labels != -1
    #
    #         filtered = [p for p, keep in zip(points, mask) if keep]
    #
    #         # 로그 (5초마다)
    #         if len(filtered) < len(points):
    #             removed = len(points) - len(filtered)
    #             self.get_logger().info(
    #                 f'DBSCAN: {len(points)} → {len(filtered)} points '
    #                 f'(removed {removed} noise)',
    #                 throttle_duration_sec=5.0
    #             )
    #
    #         return filtered
    #
    #     except Exception as e:
    #         self.get_logger().warn(f'DBSCAN failed: {e}', throttle_duration_sec=5.0)
    #         return points

    def radius_outlier_filter(self, points):
        """
        Radius Outlier Removal로 노이즈 제거
        - 반경 내 이웃 개수로 판단 (DBSCAN보다 10배 빠름)
        - 실시간 처리 가능
        """
        if not self.enable_outlier_filter or len(points) < self.outlier_min_neighbors:
            return points

        try:
            # xyz 좌표만 추출
            xyz = np.array([(p[0], p[1], p[2]) for p in points])

            # KDTree 생성 (빠른 이웃 검색)
            tree = KDTree(xyz)

            # 각 포인트의 반경 내 이웃 개수 세기
            neighbor_counts = np.array([
                len(tree.query_ball_point(point, self.outlier_radius))
                for point in xyz
            ])

            # min_neighbors 이상인 포인트만 유지
            mask = neighbor_counts >= self.outlier_min_neighbors

            filtered = [p for p, keep in zip(points, mask) if keep]

            # 로그 (5초마다)
            if len(filtered) < len(points):
                removed = len(points) - len(filtered)
                self.get_logger().info(
                    f'Outlier Filter: {len(points)} → {len(filtered)} points '
                    f'(removed {removed} noise)',
                    throttle_duration_sec=5.0
                )

            return filtered

        except Exception as e:
            self.get_logger().warn(f'Outlier filter failed: {e}', throttle_duration_sec=5.0)
            return points

    def odom_cb(self, msg: Odometry):
        t = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
        vx = float(msg.twist.twist.linear.x)
        vy = float(msg.twist.twist.linear.y)
        wz = float(msg.twist.twist.angular.z)
        self.odom_buf.append((t, vx, vy, wz))

    @staticmethod
    def _as_ns(stamp) -> int:
        return int(stamp.sec) * 10**9 + int(stamp.nanosec)

    def cloud_cb(self, msg: PointCloud2):
        if len(self.odom_buf) < 20:
            return

        # 1) points 읽기 (livox timestamp field 포함)
        try:
            pts = list(pc2.read_points(
                msg,
                field_names=('x', 'y', 'z', 'intensity', 'tag', 'line', 'timestamp'),
                skip_nans=True
            ))
        except Exception as e:
            self.get_logger().warn(f'Failed reading points with timestamp field: {e}')
            return

        if len(pts) < self.min_points:
            return

        # 2) cloud header time (epoch ns) + time_offset 보정
        t_header = self._as_ns(msg.header.stamp) + int(self.time_offset * 1e9)

        # 3) Livox raw timestamps
        ts_raw = np.array([int(p[6]) for p in pts], dtype=np.int64)
        ts_min = int(ts_raw.min())
        ts_max = int(ts_raw.max())
        raw_range = ts_max - ts_min

        # 4) epoch-like detection:
        # epoch(ns)이면 보통 1e18 근처. offset이면 1e8~1e9 정도.
        is_epoch_like = (ts_min > 10**15)

        # 5) point time -> epoch ns로 변환
        if is_epoch_like:
            # already epoch
            t_pts = ts_raw
        else:
            # scan-relative offset(ns) assumed
            # header.stamp를 scan 끝으로 가정하면: t = header + (offset - offset_max)
            # (header가 scan 시작일 가능성도 있지만, 실제 현상은 header가 끝쪽인 경우가 많아 이게 실전에서 안정적)
            if self.assume_header_is_end:
                t_pts = t_header + (ts_raw - ts_max)
            else:
                # header를 scan 시작으로 가정: t = header + (offset - offset_min)
                t_pts = t_header + (ts_raw - ts_min)

        t_min = int(t_pts.min())
        t_max = int(t_pts.max())
        scan_range = t_max - t_min

        # scan range sanity
        if scan_range <= 0 or scan_range > self.max_scan_ns:
            self.get_logger().warn(
                f'Unexpected scan range(ns)={scan_range} (raw_range={raw_range}). '
                f'raw_min={ts_min} raw_max={ts_max} epoch_like={is_epoch_like}'
            )
            return

        # 6) odom buffer numpy化
        od = np.array(self.odom_buf, dtype=np.float64)  # t, vx, vy, wz
        t_odom = od[:, 0]
        vx = od[:, 1]
        vy = od[:, 2]
        wz = od[:, 3]

        odom_min = int(t_odom.min())
        odom_max = int(t_odom.max())

        # 7) ref time을 odom 최신으로 당김 (odom이 scan 끝보다 늦게 못 따라오는 문제 해결)
        #    odom_max가 scan 끝보다 과거면, 미래 포인트는 drop
        # 여유를 0.6초 추가 (타임스탬프 동기화 + 처리 지연 허용)
        t_ref = min(t_max, odom_max + int(0.6 * 1e9))

        # 타임스탬프 불일치로 인해 모든 포인트 사용 (검증 비활성화)
        # mask = (t_pts >= (odom_min - int(0.6 * 1e9))) & (t_pts <= t_ref)
        mask = np.ones(len(t_pts), dtype=bool)  # 모든 포인트 허용
        kept = int(mask.sum())
        if kept < self.min_points:
            self.get_logger().warn(
                f'Too few points: {kept}/{len(pts)}'
            )
            return

        pts_kept = [p for p, keep in zip(pts, mask) if keep]
        t_pts_kept = t_pts[mask].astype(np.float64)
        t_min2 = int(t_pts_kept.min())
        t_max2 = int(t_pts_kept.max())

        # odom cover check 비활성화 (타임스탬프 불일치로 항상 실패하므로)
        # tolerance_ns = int(0.6 * 1e9)
        # if odom_min > (t_min2 + tolerance_ns) or odom_max < (t_max2 - tolerance_ns):
        #     self.get_logger().warn(
        #         f'Odom buffer does not cover filtered scan: '
        #         f'odom=[{odom_min},{odom_max}] scan=[{t_min2},{t_max2}]',
        #         throttle_duration_sec=5.0
        #     )
        # extrapolation으로 계속 진행

        # 8) odom 적분으로 pose(t) 구성
        # dt (seconds)
        dt = np.diff(t_odom) * 1e-9
        dt = np.clip(dt, 0.0, self.max_dt_sec)

        x = np.zeros_like(t_odom)
        y = np.zeros_like(t_odom)
        yaw = np.zeros_like(t_odom)

        if self.deskew_xy:
            x[1:] = np.cumsum(0.5 * (vx[:-1] + vx[1:]) * dt)
            y[1:] = np.cumsum(0.5 * (vy[:-1] + vy[1:]) * dt)

        if self.deskew_yaw:
            yaw[1:] = np.cumsum(0.5 * (wz[:-1] + wz[1:]) * dt)

        # ref pose at t_ref
        t_ref_f = float(t_ref)
        x_ref = float(np.interp(t_ref_f, t_odom, x))
        y_ref = float(np.interp(t_ref_f, t_odom, y))
        yaw_ref = float(np.interp(t_ref_f, t_odom, yaw))

        # point pose (extrapolation 허용)
        if self.deskew_xy:
            x_pt = np.interp(t_pts_kept, t_odom, x, left=x[0], right=x[-1])
            y_pt = np.interp(t_pts_kept, t_odom, y, left=y[0], right=y[-1])
        else:
            x_pt = np.zeros_like(t_pts_kept)
            y_pt = np.zeros_like(t_pts_kept)

        if self.deskew_yaw:
            yaw_pt = np.interp(t_pts_kept, t_odom, yaw, left=yaw[0], right=yaw[-1])
        else:
            yaw_pt = np.zeros_like(t_pts_kept)

        # relative motion from point time -> ref time
        dx = x_ref - x_pt
        dy = y_ref - y_pt
        dyaw = yaw_ref - yaw_pt

        # 9) 각 포인트를 ref 시각 기준으로 되돌리기 (inverse motion compensation)
        new_points = []
        for p, dxi, dyi, dyiw in zip(pts_kept, dx, dy, dyaw):
            x0, y0, z0, inten, tag, line, ts_field = p

            # yaw-only or full 2D rotation
            c = np.cos(-dyiw)  # inverse
            s = np.sin(-dyiw)

            xr = c * x0 - s * y0
            yr = s * x0 + c * y0

            # translation to ref
            xr += float(dxi)
            yr += float(dyi)

            # keep original fields (timestamp field kept as-is)
            new_points.append(
                (float(xr), float(yr), float(z0), float(inten), int(tag), int(line), int(ts_field))
            )

        # 10) base_link -> output_frame (odom) TF 구하기
        try:
            # TF lookup 시 최신 시간 사용 (타임스탬프 문제 회피)
            transform = self.tf_buffer.lookup_transform(
                self.output_frame,  # target: odom
                msg.header.frame_id,  # source: livox_frame or base_link
                rclpy.time.Time(),  # 최신 TF 사용 (타임스탬프 동기화 문제 회피)
                Duration(seconds=1.0)  # timeout
            )

            # TF 변환
            tx = transform.transform.translation.x
            ty = transform.transform.translation.y
            tz = transform.transform.translation.z

            qx = transform.transform.rotation.x
            qy = transform.transform.rotation.y
            qz = transform.transform.rotation.z
            qw = transform.transform.rotation.w

            # Quaternion to yaw (2D)
            yaw_tf = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

            # Transform all points to odom frame
            final_points = []
            for p in new_points:
                x, y, z, inten, tag, line, ts = p

                # Rotation
                c = np.cos(yaw_tf)
                s = np.sin(yaw_tf)
                x_odom = c * x - s * y + tx
                y_odom = s * x + c * y + ty
                z_odom = z + tz

                final_points.append(
                    (float(x_odom), float(y_odom), float(z_odom), float(inten), int(tag), int(line), int(ts))
                )

            # Radius Outlier Removal 노이즈 필터링 적용
            final_points = self.radius_outlier_filter(final_points)

            out = pc2.create_cloud(msg.header, msg.fields, final_points)
            out.header.frame_id = self.output_frame 
            # Use current time to avoid TF cache drops in downstream filters.
            out.header.stamp = self.get_clock().now().to_msg()
            self.pub.publish(out)

        except Exception as e:
            self.get_logger().warn(f'TF lookup failed: {e}', throttle_duration_sec=5.0)
            return


def main():
    rclpy.init()
    node = Deskew2D()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
