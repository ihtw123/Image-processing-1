import cv2
import numpy as np
import tkinter as tk
from tkinterdnd2 import TkinterDnD, DND_FILES  # 用于拖拽文件
from tkinter import filedialog
import threading

# 全局变量
points = []  # 存储点击的四个点
scale_factor = 1.0  # 初始缩放比例
img = None  # 原始图像
img_resized = None  # 当前缩放后的图像
warped_img = None  # 透视变换后的图像
warped_scale_factor = 1.0  # 透视变换图像的缩放比例

# 鼠标事件回调函数
def mouse_event(event, x, y, flags, param):
    global points, img, scale_factor, img_resized, warped_img, warped_scale_factor

    # 左键点击事件
    if event == cv2.EVENT_LBUTTONDOWN:
        # 根据当前缩放比例将点击点的坐标保存
        original_x = int(x / scale_factor)
        original_y = int(y / scale_factor) 
        points.append((original_x, original_y))  # 记录缩放前的坐标
        print(f"点击坐标: ({x}, {y}) -> 原始坐标: ({original_x}, {original_y})")  # 打印坐标，方便调试
        
        # 在点击的位置画一个红色圆点
        cv2.circle(img_resized, (x, y), 5, (0, 0, 255), -1)  # 红色圆点
        cv2.imshow("Image", img_resized)  # 更新图像显示

        # 如果点击了四个点，进行透视变换
        if len(points) == 4:
            order_points()

    # 鼠标滚轮事件
    if event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:
            scale_factor *= 1.1  # 放大
        else:
            scale_factor /= 1.1  # 缩小
        
        # 限制最大和最小的缩放比例
        scale_factor = max(0.1, min(scale_factor, 3.0))  # 防止过度缩放
        
        # 缩放原始图像
        img_resized = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
        
        # 调整所有点击点的位置
        img_with_points = img_resized.copy()
        for (px, py) in points:
            scaled_x = int(px * scale_factor)
            scaled_y = int(py * scale_factor)
            cv2.circle(img_with_points, (scaled_x, scaled_y), 5, (0, 0, 255), -1)
        
        # 更新图像显示
        cv2.imshow("Image", img_with_points)

        # 如果透视变换后的图像已经存在，更新其缩放
        if warped_img is not None:
            warped_img_resized = cv2.resize(warped_img, None, fx=warped_scale_factor, fy=warped_scale_factor, interpolation=cv2.INTER_LINEAR)
            cv2.imshow("Warped Image", warped_img_resized)

def order_points():
    """ 排序四个点：左上、右上、右下、左下 """
    pts = np.array(points, dtype="float32")
    
    # 排序四个点，按照顺时针顺序
    rect = order_points_clockwise(pts)
    
    # 进行透视变换
    transform_perspective(rect)

def order_points_clockwise(pts):
    """ 将四个点按照顺时针顺序排序 """
    rect = np.zeros((4, 2), dtype="float32")
    
    # 计算点的坐标总和，左上和右下分别是总和最小和最大的两个点
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # 左上
    rect[2] = pts[np.argmax(s)]  # 右下
    
    # 计算坐标差值，右上和左下分别是坐标差值最小和最大的两个点
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # 右上
    rect[3] = pts[np.argmax(diff)]  # 左下
    
    return rect

def transform_perspective(rect):
    """ 进行透视变换 """
    global warped_img, warped_scale_factor
    # 获取矩形的四个角
    (top_left, top_right, bottom_right, bottom_left) = rect
    width_a = np.linalg.norm(bottom_right - bottom_left)
    width_b = np.linalg.norm(top_right - top_left)
    height_a = np.linalg.norm(top_right - bottom_right)
    height_b = np.linalg.norm(top_left - bottom_left)

    # 选择最大宽度和高度
    max_width = max(int(width_a), int(width_b))
    max_height = max(int(height_a), int(height_b))
    
    # 目标矩阵的四个顶点
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")
    
    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(rect, dst)
    
    # 执行透视变换
    warped_img = cv2.warpPerspective(img, M, (max_width, max_height))
    
    # 获取屏幕分辨率
    screen_width = 1200  # 例如 1920x1080 屏幕
    screen_height = 900
    
    # 缩放透视变换后的图像适应屏幕大小
    warped_scale_factor = min(screen_width / max_width, screen_height / max_height) * 0.9  # 适当缩放
    warped_img_resized = cv2.resize(warped_img, None, fx=warped_scale_factor, fy=warped_scale_factor, interpolation=cv2.INTER_LINEAR)
    
    # 显示透视变换后的图像
    cv2.imshow("Warped Image", warped_img_resized)
    
    # 透视变换图像窗口已显示，可以设置回调
    cv2.setMouseCallback("Warped Image", on_warped_image_scroll)

def on_warped_image_scroll(event, x, y, flags, param):
    global warped_scale_factor, warped_img

    if event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:
            warped_scale_factor *= 1.1  # 放大
        else:
            warped_scale_factor /= 1.1  # 缩小
        
        # 限制最大和最小的缩放比例
        warped_scale_factor = max(0.1, min(warped_scale_factor, 3.0))  # 防止过度缩放
        
        # 缩放透视变换后的图像
        warped_img_resized = cv2.resize(warped_img, None, fx=warped_scale_factor, fy=warped_scale_factor, interpolation=cv2.INTER_LINEAR)
        cv2.imshow("Warped Image", warped_img_resized)

# 读取图片并调整大小
def load_image_from_file(file_path):
    global img, img_resized, scale_factor
    img = cv2.imread(file_path)
    if img is None:
        print("图像加载失败，请检查路径")
        exit()  # 如果没有加载图像，程序退出

    # 获取屏幕分辨率
    screen_width = 1920  # 例如 1920x1080 屏幕
    screen_height = 1080

    # 计算适应屏幕大小的缩放比例
    height, width = img.shape[:2]
    scale_factor = min(screen_width / width, screen_height / height) * 0.9  # 设置为适当的缩放比例，使图片不至于过大

    # 缩放图像以适应屏幕
    img_resized = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

    # 显示图像并设置回调函数
    cv2.imshow("Image", img_resized)
    cv2.setMouseCallback("Image", mouse_event)

# 使用 tkinter 实现拖拽文件功能
def on_drop(event):
    file_path = event.data
    print(f"拖拽文件路径: {file_path}")
    load_image_from_file(file_path)

def opencv_loop():
    # 持续监听图像窗口事件
    while True:
        key = cv2.waitKey(1)  # 每1毫秒检查一次用户输入
        if key == ord('q'):
            break
    cv2.destroyAllWindows()

def exit_program():
    """ 用于关闭 GUI 和 OpenCV 窗口 """
    cv2.destroyAllWindows()
    root.quit()

def on_resize(event):
    """ 窗口大小变化时，不重新调整图像大小 """
    pass  # 不再处理窗口大小变化事件

# 重新绘制上一个点
def reset_last_point():
    global points, img_resized
    if points:
        points.pop()  # 移除最后一个点
    
    # 重新绘制图像并清除所有红点
    img_with_points = img.copy()  # 使用原始图像进行清除
    img_resized = cv2.resize(img_with_points, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
    
    # 重新绘制当前的点
    for (px, py) in points:
        scaled_x = int(px * scale_factor)
        scaled_y = int(py * scale_factor)
        cv2.circle(img_resized, (scaled_x, scaled_y), 5, (0, 0, 255), -1)
    
    # 显示图像
    cv2.imshow("Image", img_resized)

# 重新绘制四个点
def reset_all_points():
    global points, img_resized
    points = []  # 清空原来的点
    
    # 重新生成原始图像并清除红点
    img_with_points = img.copy()  # 使用原始图像
    img_resized = cv2.resize(img_with_points, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
    
    # 重新绘制当前的点
    for (px, py) in points:
        scaled_x = int(px * scale_factor)
        scaled_y = int(py * scale_factor)
        cv2.circle(img_resized, (scaled_x, scaled_y), 5, (0, 0, 255), -1)
    
    # 显示图像
    cv2.imshow("Image", img_resized)

# 保存透视变换后的图像
def save_image():
    global warped_img, warped_scale_factor
    if warped_img is not None:
        # 打开文件保存对话框
        save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")])
        
        if save_path:
            # 保存透视变换后的图像
            cv2.imwrite(save_path, warped_img)
            print(f"透视变换后的图像已保存到: {save_path}")
    else:
        print("没有可保存的透视变换图像。")

# 设置 GUI
def main():
    global root
    root = TkinterDnD.Tk()
    root.title("图像透视变换")
    root.geometry("800x600")

    # 允许窗口调整大小
    root.resizable(True, True)

    # 创建拖放区
    label = tk.Label(root, text="请将图像拖拽到此窗口", width=50, height=10)
    label.pack(padx=10, pady=10)

    # 设置文件拖拽回调
    label.drop_target_register(DND_FILES)
    label.dnd_bind('<<Drop>>', on_drop)

    # 设置窗口大小变化回调，空函数，防止调整窗口大小时图像变动
    root.bind("<Configure>", on_resize)

    # 创建按钮
    reset_point_button = tk.Button(root, text="重新画上一个点", command=reset_last_point)
    reset_point_button.pack(pady=10)

    reset_all_points_button = tk.Button(root, text="重新画四个点", command=reset_all_points)
    reset_all_points_button.pack(pady=10)

    # 创建保存按钮
    save_button = tk.Button(root, text="保存处理后的图像", command=save_image)
    save_button.pack(pady=10)

    # 添加退出按钮
    exit_button = tk.Button(root, text="退出", command=exit_program)
    exit_button.pack(pady=20)

    # 启动 OpenCV 事件循环线程
    opencv_thread = threading.Thread(target=opencv_loop)
    opencv_thread.daemon = True
    opencv_thread.start()

    # 启动 GUI 主事件循环
    root.mainloop()

if __name__ == "__main__":
    main()
