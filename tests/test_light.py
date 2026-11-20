import cv2
import argparse

def adjust_video_brightness(input_path, output_path, beta=15):
    """
    调节视频亮度
    :param input_path: 输入视频路径
    :param output_path: 输出视频路径
    :param beta: 亮度偏移量，正数变亮，负数变暗，推荐 5~30
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"无法打开视频：{input_path}")
        return

    # 获取视频参数
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"开始处理，beta={beta}，分辨率：{width}x{height}, fps={fps:.2f}")
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 亮度调整：src + beta，自动截断到0~255，防止溢出
        frame = cv2.add(frame, beta)
        out.write(frame)

        frame_count += 1
        if frame_count % 100 == 0:
            print(f"已处理 {frame_count} 帧")

    cap.release()
    out.release()
    print(f"处理完成！共 {frame_count} 帧，输出：{output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="视频亮度提升工具")
    parser.add_argument("-i", "--input", required=True, help="输入视频路径")
    parser.add_argument("-o", "--output", required=True, help="输出视频路径")
    parser.add_argument("-b", "--beta", type=int, default=15, help="亮度值，默认15，范围建议 -50 ~ 50")
    args = parser.parse_args()

    adjust_video_brightness(args.input, args.output, args.beta)
