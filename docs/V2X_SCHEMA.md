# V2X 运动目标状态消息 schema（本项目实现版）

## 消息（每帧一条）
| 字段 | 类型 | 说明 |
|---|---|---|
| station_id | str | 感知站标识（车载 `VEH-EGO-001` / 路侧 `RSU-001`…） |
| frame | int | 源帧号（BDD 为 5 fps 序列帧号） |
| ts | float | 时间戳(秒) = frame / fps |
| ttl | float | 消息有效期(秒)，超过即视为过期目标 |
| n_obj | int | 本帧目标数 |
| objects | list | 目标状态列表 |

## 目标（objects[i]）
| 字段 | 类型 | 单位 | 说明 |
|---|---|---|---|
| track_id | int | — | 跟踪器维持的连续 ID（**ID 维持率的载体**） |
| cls / cls_code | str/int | — | 类别名与编码（1 ped, 2 rider, 3 car, 4 truck, 5 bus, 6 train, 7 moto, 8 bike） |
| bbox_px | float[4] | px | x1,y1,x2,y2 |
| center_px | float[2] | px | 框中心 |
| center_norm | float[2] | [0,1] | 归一化中心，跨分辨率可用 |
| size_px | float[2] | px | 宽、高 |
| vel_px_s | float[2] | px/s | 图像平面速度（已按 fps 换算） |
| heading_deg | float/null | deg | 运动方向，0=图像正上（远方），顺时针为正 |
| conf | float | — | 检测置信度 |
| state | str | — | tracked / lost |
| pos_m_est | float[2] | m | **可选**：假定焦距 + 地面平面下的粗略米制位置（X 右, Z 前），仅供演示 |
| speed_mps_est | float | m/s | **可选**：由米制位置推得的速度粗估 |

## 关键说明（写报告时必须保留）
- BDD100K 车载相机**无标定参数**，`pos_m_est` / `speed_mps_est` 基于假设（f=1108 px, 相机高 1.4 m, 地面平面），
  只能用于演示消息结构，**不能作为精度结论**。
- 若要真实米制轨迹，需改用带标定的数据集（KITTI Tracking 的 `calib` + 3D 真值）。
