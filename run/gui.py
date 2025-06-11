import os
import sys
import torch
import pickle  # 用于加载 BERT 分词器
import tempfile  # 用于为 PredictionWorker 创建临时目录
import numpy as np  # 用于结果显示中的 numpy 数组转换

from PySide6.QtWidgets import (QApplication, QMainWindow, QTextEdit, QPushButton,
                               QLabel, QVBoxLayout, QWidget, QFileDialog, QMessageBox,
                               QProgressBar, QHBoxLayout, QGraphicsDropShadowEffect,
                               QSizePolicy, QScrollArea, QFrame, QComboBox)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap, QColor, QPalette, QBrush

# 导入后端组件
# 确保 backend_cn.py 与 gui_cn.py 在同一目录或 Python 的 path 中
try:
    from backend import PredictionWorker, MultiModalEfficientNet, EfficientNetB1_1D, EfficientNet1D, KmerBERT
except ImportError as e:
    # 提供更详细的错误信息，帮助用户定位问题
    current_dir = os.path.dirname(os.path.abspath(__file__))
    backend_path = os.path.join(current_dir, "backend.py")
    error_msg = (f"错误：无法导入后端组件: {e}\n"
                 f"请确保 'backend.py' 文件存在于以下目录中:\n{current_dir}\n"
                 f"或者 'backend.py' 所在的目录已添加到您的 PYTHONPATH 环境变量中。\n"
                 f"尝试查找路径: {backend_path}")
    # 在尝试显示GUI之前，通过控制台打印错误并退出可能更安全
    print(error_msg, file=sys.stderr)
    # 对于GUI应用，也可以尝试用QMessageBox显示，但若QApplication未初始化则不行
    # app_temp = QApplication.instance() # 检查是否有实例
    # if not app_temp: app_temp = QApplication(sys.argv) # 创建一个临时实例来显示消息
    # QMessageBox.critical(None, "导入错误", error_msg)
    sys.exit(f"关键错误：无法加载后端模块。详细信息：{error_msg}")


class ModernWindow(QMainWindow):
    """
    RNA 分类器应用程序的主窗口。
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("基于Bert的RNA分类器")  # 窗口标题
        self.resize(1280, 720)  # 初始尺寸
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setMinimumSize(800, 500)
        self.setMaximumSize(2560, 1600)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"正在使用设备: {self.device}")
        # 模型路径
        self.model_paths = {
            "EIIP": "../model/best_model_ilearn.pth",
            "FOLD": "../model/best_model_fold.pth",
            "BERT": "../model/kmer_bert_model.pth",
            "HYBRID": "../model/best_model_hybird509.pth"
        }
        self.tokenizer_path = "../model/tokenizer.pkl"  # BERT分词器路径

        self.model = None  # 将持有当前加载的模型
        self.tokenizer = None  # 将持有 BERT 的分词器

        # 用于预测产物的临时目录管理器
        self.prediction_temp_dir_manager = None

        self.setup_background()  # 设置背景
        self.setup_central_widget()  # 设置中央面板
        self.setup_ui()  # 在面板内设置主要UI元素
        self.load_model()  # 根据 ComboBox 中的默认选项初始加载模型

    def setup_background(self):
        self.background_label = QLabel(self)
        self.background_label.setGeometry(0, 0, self.width(), self.height())
        bg_pixmap = QPixmap("back.png")  # 确保 back.png 在执行目录下

        if bg_pixmap.isNull():
            print("警告: 背景图片 'back.png' 未找到。将使用纯色背景。")
            palette = self.palette()
            palette.setColor(QPalette.ColorRole.Window, QColor(173, 216, 230))
            self.setPalette(palette)
            self.setAutoFillBackground(True)
        else:
            # 强制拉伸填充（忽略图片比例）
            scaled_pixmap = bg_pixmap.scaled(self.size(), Qt.AspectRatioMode.IgnoreAspectRatio,
                                             Qt.TransformationMode.SmoothTransformation)
            self.background_label.setPixmap(scaled_pixmap)
            self.background_label.setScaledContents(True)  # 允许内容缩放
        self.background_label.lower()  # 确保背景在最底层

    def resizeEvent(self, event):
        """处理窗口大小调整事件，保持宽高比（如果原始逻辑如此）并更新UI。"""
        current_size = self.size()
        width, height = current_size.width(), current_size.height()

        # 原始代码中的16:9比例调整逻辑
        target_width = width
        target_height = int(width * 9 / 16)
        if abs(target_height - height) > 20:  # 阈值判断
            target_height = height
            target_width = int(height * 16 / 9)

        # 这样做是为了避免在某些平台上可能发生的无限递归调整
        if (target_width != width or target_height != height) and \
                (abs(target_width - width) > 1 or abs(target_height - height) > 1):
            self.blockSignals(True)  # 阻止信号，避免可能的递归调用或性能问题
            self.resize(target_width, target_height)
            self.blockSignals(False)
            # 更新 width 和 height 以反映新的窗口尺寸
            current_size = self.size()  # 获取调整后的尺寸
            width, height = current_size.width(), current_size.height()

        # 更新背景图片大小
        if hasattr(self,
                   'background_label') and self.background_label.pixmap() and not self.background_label.pixmap().isNull():
            bg_pixmap_orig = QPixmap("back.png")  # 重新加载原始图片以获得最佳缩放质量
            if not bg_pixmap_orig.isNull():
                scaled_pixmap = bg_pixmap_orig.scaled(width, height, Qt.AspectRatioMode.IgnoreAspectRatio,
                                                      Qt.TransformationMode.SmoothTransformation)
                self.background_label.setPixmap(scaled_pixmap)
            self.background_label.resize(width, height)  # 确保标签尺寸也更新

        # 动态调整面板尺寸和位置
        if hasattr(self, 'panel'):
            panel_width_ratio = 0.71  # 原始比例
            panel_height_to_width_ratio = 0.73  # 原始比例：面板高度是其自身宽度的0.73

            calculated_panel_width = int(width * panel_width_ratio)
            calculated_panel_height = int(calculated_panel_width * panel_height_to_width_ratio)

            # 确保面板在窗口内并居中
            panel_x = (width - calculated_panel_width) // 2
            panel_y = (height - calculated_panel_height) // 2
            self.panel.setGeometry(panel_x, panel_y, calculated_panel_width, calculated_panel_height)

            # 调用原始的UI元素更新逻辑
            self._update_ui_elements_original_logic(calculated_panel_width, calculated_panel_height)
            if self.panel.layout():
                self.panel.layout().update()  # 强制刷新布局

        super().resizeEvent(event)  # 调用基类实现

    def _update_ui_elements_original_logic(self, panel_width, panel_height):
        """根据面板尺寸动态调整UI元素"""
        if not hasattr(self, 'panel') or not self.panel.layout():
            return

        main_layout = self.panel.layout()  # 即 panel_main_layout (QHBoxLayout)

        # 动态计算边距和间距
        padding = max(30, int(panel_width * 0.06))
        spacing = max(20, int(panel_width * 0.03))  # 这个spacing是针对main_layout的子布局的，不是main_layout自身
        main_layout.setContentsMargins(padding, padding, padding, padding)
        # main_layout.setSpacing(spacing) # QHBoxLayout的setSpacing

        # 动态标题样式
        if hasattr(self, 'title_label'):
            font_size_title = max(24, int(panel_width * 0.03))  # 原始计算
            self.title_label.setStyleSheet(f"""
                font-size: {font_size_title}px;
                font-weight: bold;
                color: {self.color_scheme['primary']};
            """)
            # 它的间距也可能需要调整，或者其父布局 (left_layout) 的间距
            if hasattr(self, 'left_layout'):  # 假设 title_container 在 left_layout 中
                self.left_layout.setSpacing(max(15, int(panel_height * 0.03)))  # 动态调整left_layout的间距

        # 动态按钮样式
        if hasattr(self, 'predict_btn'):
            btn_font_size = max(14, int(panel_width * 0.025))  # 原始计算
            btn_padding = int(btn_font_size * 0.8)  # 原始计算
            btn_radius = int(btn_font_size * 0.8)  # 原始计算
            self.predict_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {self.color_scheme['primary']};
                    color: white;
                    padding: {btn_padding}px;
                    border-radius: {btn_radius}px;
                    font-size: {btn_font_size}px;
                    font-weight: bold;
                    border: none;
                }}
                QPushButton:hover {{ background-color: {self.color_scheme['primary_light']}; }}
                QPushButton:pressed {{ background-color: #1a252f; }}
            """)

        # 动态输入框样式
        if hasattr(self, 'seq_input') and hasattr(self, 'id_input'):
            font_size_input = max(14, int(panel_width * 0.02))  # 原始计算
            padding_input = max(12, int(panel_width * 0.015))  # 原始计算
            radius_input = max(10, int(panel_width * 0.012))  # 原始计算
            input_style = f"""
                QTextEdit {{
                    padding: {padding_input}px;
                    border: 2px solid {self.color_scheme['border']};
                    border-radius: {radius_input}px;
                    font-size: {font_size_input}px;
                    background-color: {self.color_scheme['background']};
                    color: {self.color_scheme['text']};
                }}
                QTextEdit:focus {{
                    border: 2px solid {self.color_scheme['primary']};
                    background-color: white;
                }}
            """
            self.seq_input.setStyleSheet(input_style)
            self.id_input.setStyleSheet(input_style)

        # 强制刷新布局
        main_layout.activate()

    def setup_central_widget(self):
        """设置主面板控件，该控件包含所有UI内容"""
        self.panel = QWidget(self)
        self.panel.setObjectName("mainPanel")
        # 面板将在 resizeEvent 中定位和调整大小，或在此处进行初始设置
        # 初始面板尺寸和位置（在第一次 resizeEvent 调用前）
        initial_width = self.width()
        initial_height = self.height()
        panel_w = int(initial_width * 0.71)
        panel_h = int(panel_w * 0.73)
        panel_x_init = (initial_width - panel_w) // 2
        panel_y_init = (initial_height - panel_h) // 2
        self.panel.setGeometry(panel_x_init, panel_y_init, panel_w, panel_h)

        self.panel.setStyleSheet("""
            #mainPanel {
                background-color: rgba(255, 255, 255, 200); /* 半透明白色 */
                border-radius: 20px; /* 圆角 */
            }
        """)
        # 阴影效果
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(25)
        shadow.setColor(QColor(0, 0, 0, 100))
        shadow.setOffset(0, 5)
        self.panel.setGraphicsEffect(shadow)

        # 面板配色方案
        self.color_scheme = {
            'primary': '#2c3e50',
            'primary_light': '#34495e',
            'accent': '#3498db',
            'text': '#1a1a1a',
            'border': '#7f8c8d',
            'background': 'rgba(245, 247, 250, 220)'
        }

    def get_input_style_from_original(self):
        # 这是基于原始代码的静态版本，panel_width 在调用时可能还未最终确定
        # 因此，使用一个合理的默认值或在 _update_ui_elements_original_logic 中动态设置
        return f"""
            QTextEdit {{
                padding: 12px; 
                border: 2px solid {self.color_scheme['border']};
                border-radius: 12px; /* 原始是12px */
                font-size: 10px; /* 原始是10px, 在动态调整中会变大 */
                background-color: {self.color_scheme['background']};
                color: {self.color_scheme['text']};
            }}
            QTextEdit:focus {{
                border: 2px solid {self.color_scheme['primary']};
                background-color: white;
            }}
        """

    def get_button_style_from_original(self):
        """复现原始 GUI.txt 中的 get_button_style()，用于初始设置"""
        return f"""
            QPushButton {{
                background-color: {self.color_scheme['primary']};
                color: white;
                padding: 14px;
                border-radius: 12px; /* 原始是12px */
                font-size: 16px; /* 原始是16px */
                font-weight: bold;
                border: none;
            }}
            QPushButton:hover {{ background-color: {self.color_scheme['primary_light']}; }}
            QPushButton:pressed {{ background-color: #1a252f; }}
        """

    def setup_ui(self):
        """在主面板内设置用户界面元素 """
        # 面板的主布局 (QHBoxLayout 用于左右区域)
        self.panel_main_layout = QHBoxLayout(self.panel)
        # 边距和间距将在 _update_ui_elements_original_logic 中动态设置
        # 初始边距和间距
        self.panel_main_layout.setContentsMargins(50, 50, 50, 30)
        self.panel_main_layout.setSpacing(30)

        # --- 左侧区域 (输入) ---
        left_widget = QWidget()
        self.left_layout = QVBoxLayout(left_widget)  # 保存为成员变量以便动态调整
        self.left_layout.setSpacing(25)  # 原始间距

        # 标题行容器 (标题 + 图标)
        title_container = QHBoxLayout()
        title_container.setSpacing(15)  # 原始间距

        self.title_label = QLabel("RNA-Classifier")  # 标题文本，与原始一致
        # 初始样式，动态调整见 _update_ui_elements_original_logic
        self.title_label.setStyleSheet(f"""
            font-size: 32px; font-weight: bold; color: {self.color_scheme['primary']};
        """)

        self.title_icon = QLabel()
        icon_pixmap = QPixmap("label.png")  # 确保此图标可用
        if not icon_pixmap.isNull():
            # 原始缩放到高度90，这里保持一致
            scaled_icon = icon_pixmap.scaledToHeight(90, Qt.TransformationMode.SmoothTransformation)
            self.title_icon.setPixmap(scaled_icon)
        else:
            print("警告: 标题图标 'label.png' 未找到。")
        # 原始图标样式
        self.title_icon.setStyleSheet("background-color: rgba(255, 255, 255, 10); padding: 3px;")
        icon_shadow = QGraphicsDropShadowEffect(self)
        icon_shadow.setBlurRadius(5);
        icon_shadow.setColor(QColor(0, 0, 0, 20));
        icon_shadow.setOffset(0, 1)
        self.title_icon.setGraphicsEffect(icon_shadow)

        title_container.addWidget(self.title_label)
        title_container.addWidget(self.title_icon)
        title_container.addStretch()  # 防止内容压缩
        self.left_layout.addLayout(title_container)

        # RNA 序列输入
        seq_prompt_label = QLabel("输入RNA序列:")  # 提示标签
        seq_prompt_label.setStyleSheet("color: #8c9ba4;font-size: 17px;")  # 原始样式
        self.left_layout.addWidget(seq_prompt_label)

        self.seq_input = QTextEdit()
        self.seq_input.setPlaceholderText("请输入RNA序列（例如：AUCCGCCGCCGU...）")  # 占位符文本
        self.seq_input.setStyleSheet(self.get_input_style_from_original())  # 初始样式
        self.left_layout.addWidget(self.seq_input)  # 添加RNA序列输入框
        # 序列 ID 输入
        id_prompt_label = QLabel("输入序列ID:")  # 提示标签
        id_prompt_label.setStyleSheet("color: #8c9ba4;font-size: 17px;")  # 原始样式
        self.left_layout.addWidget(id_prompt_label)

        self.id_input = QTextEdit()
        self.id_input.setPlaceholderText("请输入序列标识符（例如：ENST00000702787.2）")  # 占位符文本
        self.id_input.setStyleSheet(self.get_input_style_from_original())  # 初始样式
        self.left_layout.addWidget(self.id_input)  # 添加序列ID输入框

        # 预测按钮
        self.predict_btn = QPushButton("开始预测")  # 按钮文本
        self.predict_btn.setStyleSheet(self.get_button_style_from_original())  # 初始样式
        self.predict_btn.clicked.connect(self.start_prediction)
        self.left_layout.addWidget(self.predict_btn)

        # 状态区域 (标签 + 进度条)
        status_layout = QHBoxLayout()
        self.status_label = QLabel("就绪")  # 初始状态文本
        self.status_label.setStyleSheet(f"color: {self.color_scheme['text']}; font-size: 14px;")  # 与之前版本协调的样式

        self.progress_bar = QProgressBar()
        # 进度条样式，严格按照原始 GUI.txt
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                border: 2px solid {self.color_scheme['border']};
                border-radius: 2px;
                text-align: center;
                font-size: 15px;
                font-weight: bold;
                color: #000000; /* 黑色文本 */
                height: 25px;
                background-color: {self.color_scheme['background']};
            }}
            QProgressBar::chunk {{
                background-color: #FF4081; /* 粉色进度条块 */
                border-radius: 0px; /* 原始为0px */
                width: 30px; /* 原始设定，但通常Qt会根据值自动计算块 */
            }}
        """)
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.progress_bar, 1)  # 进度条占据可用空间
        self.left_layout.addLayout(status_layout)

        # --- 右侧区域 (输出和模型选择) ---
        right_widget = QWidget()
        self.right_layout = QVBoxLayout(right_widget)  # 保存为成员变量
        self.right_layout.setSpacing(15)  # 原始间距

        # 模型选择下拉框 (置于右侧面板顶部)
        model_select_container = QHBoxLayout()  # 使用 QHBoxLayout 容纳标签和下拉框
        model_select_label = QLabel("选择模型:")  # 标签文本
        model_select_label.setStyleSheet("color: #8c9ba4;font-size: 17px;")  # 原始样式
        model_select_container.addWidget(model_select_label)

        self.model_combo = QComboBox()
        self.model_combo.addItems(["EIIP", "FOLD", "BERT", "HYBRID"])  # 模型选项
        self.model_combo.setCurrentText("HYBRID")  # 默认选中 HYBRID
        # 下拉框样式，严格按照原始 GUI.txt
        self.model_combo.setStyleSheet("""
           QComboBox {
               padding: 8px;
               border: 2px solid #7f8c8d;
               border-radius: 10px;
               font-size: 14px;
               background-color: rgba(245, 247, 250, 220);
               color: #1a1a1a;
           }
           QComboBox::drop-down { border: none; }
           QComboBox::down-arrow { image: url(dropdown_icon.png); /* 可选：自定义箭头图标, 需提供图片 */ }
           QComboBox QAbstractItemView { /* 下拉列表项目的样式 */
                background-color: white; /* 白色背景 */
                border: 1px solid #7f8c8d; /* 边框 */
                selection-background-color: #3498db; /* 选中项背景色 (accent color) */
                padding: 5px; /* 内边距 */
           }
        """)
        self.model_combo.currentIndexChanged.connect(self.on_model_selected)
        model_select_container.addWidget(self.model_combo, 1)  # 下拉框占据可用空间
        self.right_layout.insertLayout(0, model_select_container)  # 插入到右侧布局顶部

        # 结果区域标签
        result_area_prompt_label = QLabel("预测结果:")  # 标签文本
        result_area_prompt_label.setStyleSheet("color: #8c9ba4;font-size: 20px;")  # 原始样式
        self.right_layout.addWidget(result_area_prompt_label)

        # 结果显示 (可滚动)
        # 代码中 result_container 和 result_layout 的结构是为了滚动条
        # 这里直接将 result_label 放入 QScrollArea
        self.result_label = QLabel("预测结果将显示在这里")  # 初始文本
        self.result_label.setWordWrap(True)  # 自动换行
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)  # 顶部左对齐
        # 结果标签样式，严格按照原始 GUI.txt
        self.result_label.setStyleSheet(f"""
            QLabel {{
                color: {self.color_scheme['text']};
                font-size: 16px;
                padding: 15px;
                background-color: rgba(245, 247, 250, 180); /* 原始背景色 */
                border-radius: 10px; /* 原始圆角 */
                min-height: 120px; /* 原始最小高度 */
            }}
        """)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)  # 允许内部控件调整大小
        scroll_area.setWidget(self.result_label)  # 将结果标签放入滚动区域
        # 滚动区域样式
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent; /* 使其透明以显示 result_label 的背景 */
            }
            QScrollBar:vertical {
                border: 1px solid #dcdcdc; background: #f0f0f0;
                width: 15px; margin: 0px 0px 0px 0px; border-radius: 7px;
            }
            QScrollBar::handle:vertical {
                background: #c0c0c0; min-height: 20px; border-radius: 7px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                border: none; background: none; height: 0px;
            }
        """)
        self.right_layout.addWidget(scroll_area, 1)  # 滚动区域占据可用垂直空间

        # --- 组装主面板布局 ---
        self.panel_main_layout.addWidget(left_widget, 2)  # 左侧区域占据2/3空间

        separator = QFrame()  # 分隔线
        separator.setFrameShape(QFrame.Shape.VLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        separator.setStyleSheet("color: #bdc3c7;")  # 原始分隔线颜色
        self.panel_main_layout.addWidget(separator)

        self.panel_main_layout.addWidget(right_widget, 1)  # 右侧区域占据1/3空间

        # 初始动态UI更新调用，以应用基于初始面板尺寸的样式
        # 确保在调用前 panel 已经有尺寸
        QTimer.singleShot(0, lambda: self._update_ui_elements_original_logic(self.panel.width(), self.panel.height()))

    def on_model_selected(self):
        """处理 QComboBox 中的模型选择更改。"""
        self.load_model()

    def load_model(self):
        """加载选定的机器学习模型和分词器 (如果是BERT)。"""
        selected_model_type = self.model_combo.currentText()
        model_path = self.model_paths.get(selected_model_type)

        if not model_path:
            self.show_error_message(f"模型 '{selected_model_type}' 的路径未定义。")
            self.model = None;
            self.tokenizer = None
            self.status_label.setText(f"错误: 模型路径缺失 ({selected_model_type})")
            return

        if not os.path.exists(model_path):
            self.show_error_message(f"模型文件未找到: {model_path}\n请确保模型位于正确的 '../model/' 目录下。")
            self.model = None;
            self.tokenizer = None
            self.status_label.setText(f"错误: 模型文件缺失 ({selected_model_type})")
            return

        self.status_label.setText(f"正在加载 {selected_model_type} 模型...")
        QApplication.processEvents()  # 更新UI以显示加载状态

        try:
            self.model = None  # 清除先前的模型
            self.tokenizer = None  # 清除先前的分词器

            if selected_model_type == "HYBRID":
                self.model = MultiModalEfficientNet()
            elif selected_model_type == "BERT":
                if not os.path.exists(self.tokenizer_path):
                    self.show_error_message(f"BERT分词器未找到: {self.tokenizer_path}")
                    self.status_label.setText("错误: BERT分词器缺失。")
                    return
                with open(self.tokenizer_path, 'rb') as f:
                    self.tokenizer = pickle.load(f)
                vocab_size = len(self.tokenizer)
                self.model = KmerBERT(vocab_size=vocab_size, num_classes=2)  # 假设2个类别
            elif selected_model_type == "FOLD":
                self.model = EfficientNet1D(num_classes=2)  # 假设2个类别
            elif selected_model_type == "EIIP":
                self.model = EfficientNetB1_1D(num_classes=1)  # EIIP模型为二分类输出1个logit

            if self.model:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.to(self.device)
                self.model.eval()
                self.status_label.setText(f"{selected_model_type} 模型加载成功。")
            else:
                self.show_error_message(f"初始化模型 '{selected_model_type}' 失败。")
                self.status_label.setText(f"加载 {selected_model_type} 出错。")

        except FileNotFoundError as fnf_err:
            self.show_error_message(f"模型加载期间文件未找到: {str(fnf_err)}")
            self.model = None;
            self.tokenizer = None
            self.status_label.setText(f"加载 {selected_model_type} 出错。")
        except Exception as e:
            self.show_error_message(f"加载 {selected_model_type} 模型失败: {str(e)}")
            self.model = None;
            self.tokenizer = None
            self.status_label.setText(f"加载 {selected_model_type} 出错。")

    def start_prediction(self):
        """启动RNA序列预测过程。"""
        rna_seq = self.seq_input.toPlainText().strip()
        seq_id = self.id_input.toPlainText().strip()

        if not rna_seq:
            self.show_error_message("RNA序列不能为空。")
            return
        if not seq_id:
            self.show_error_message("序列ID不能为空。")
            return

        if not self.model:
            self.show_error_message("模型未加载。请选择一个模型并等待其加载。")
            self.load_model()  # 尝试重新加载
            if not self.model:
                self.show_error_message("模型加载失败，无法开始预测。")
                return
            QMessageBox.information(self, "模型已加载", "模型现已加载。请再次点击“开始预测”。")
            return

        self.predict_btn.setEnabled(False)
        self.result_label.setText("正在处理，请稍候...")  # 处理中的提示
        self.progress_bar.setValue(0)
        self.status_label.setText("开始预测...")

        try:
            # 为此预测运行创建一个新的临时目录
            self.prediction_temp_dir_manager = tempfile.TemporaryDirectory()
            temp_dir_path = self.prediction_temp_dir_manager.name
        except Exception as e:
            self.show_error_message(f"创建临时目录失败: {e}")
            self.predict_btn.setEnabled(True)
            return

        current_model_type = self.model_combo.currentText()

        self.worker = PredictionWorker(
            model=self.model,
            rna_sequence=rna_seq,
            file_id=seq_id,
            temp_dir_path=temp_dir_path,
            device=self.device,
            model_type=current_model_type,
            tokenizer=self.tokenizer if current_model_type == "BERT" else None
        )

        self.worker.progress.connect(self.update_status_and_progress)
        self.worker.result_ready.connect(self.display_prediction_result)
        self.worker.error_occurred.connect(self.handle_prediction_error)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.start()

    def update_status_and_progress(self, message):
        """更新状态标签和进度条。"""
        self.status_label.setText(message)
        current_progress = self.progress_bar.value()
        # 更新逻辑是每次增加20，这里模拟类似行为
        if "第一步" in message or "Step 1" in message:
            self.progress_bar.setValue(20)
        elif "第二步" in message or "Step 2" in message:
            self.progress_bar.setValue(40)
        elif "第三步" in message or "Step 3" in message:
            self.progress_bar.setValue(60)
        elif "第四步" in message or "Step 4" in message:
            self.progress_bar.setValue(80)
        elif "第五步" in message or "Step 5" in message:
            self.progress_bar.setValue(90)
        # QTimer.singleShot(300, lambda: self.progress_bar.setValue(self.progress_bar.value() + 20)) # 原始逻辑

    def display_prediction_result(self, result_data):
        """在GUI中显示预测结果 - 严格按照原始 GUI.txt 的结果格式。"""
        self.progress_bar.setValue(100)
        self.status_label.setText("预测完成！")

        seq_id_display = self.id_input.toPlainText().strip()
        model_type_display = self.model_combo.currentText()

        pred_class_idx = result_data['class']
        probabilities_list = result_data['probabilities']  # 预期: [prob_class_0, prob_class_1]
        raw_model_output = result_data['raw_output']

        # 根据原始 GUI.txt 的逻辑确定类别名称
        # 类别 1: 非编码RNA, 类别 0: 蛋白质编码RNA
        class_name_display = '非编码RNA' if pred_class_idx == 1 else '蛋白质编码RNA'

        prob_non_coding_rna = probabilities_list[1] if len(probabilities_list) > 1 else "N/A"
        prob_coding_rna = probabilities_list[0] if len(probabilities_list) > 0 else "N/A"

        # 构建与原始 GUI.txt 中 show_result 方法一致的 HTML 文本
        # 注意 EIIP 模型有特殊处理，但这里我们先用通用格式，如果需要，可以再细化
        # 原始代码中，EIIP的概率显示顺序是编码在前，非编码在后，即使pred_class=1
        # 我们这里统一按 编码 (class 0) 和 非编码 (class 1) 的顺序展示概率

        result_html_text = f"""
        <h3 style='color: {self.color_scheme['primary']};'>{model_type_display} 模型预测结果:</h3>
        <p>RNA序列ID: <b>{seq_id_display}</b></p>
        <p>预测类别: <b>{class_name_display}</b></p>
        """

        if model_type_display == "EIIP":
            # EIIP 的概率在原始代码中是特殊显示的
            # prob_non_coding 是 class 1 (非编码) 的概率
            # prob_coding 是 1 - prob_non_coding (编码) 的概率
            # 但原始 result_dict 的 probabilities 已经是 [prob_coding, prob_non_coding]
            result_html_text += f"""
            <p>概率:</p>
            <ul>
                <li>编码RNA: {float(prob_coding_rna):.4f}</li>
                <li>非编码RNA: {float(prob_non_coding_rna):.4f}</li>
            </ul>
            <p>原始输出值: {np.array2string(raw_model_output, formatter={'float_kind': lambda x: "%.4f" % x})}</p>
            """
        else:  # BERT, FOLD, HYBRID
            result_html_text += f"""
            <p>类别概率:</p>
            <ul>
                <li>非编码RNA: {float(prob_non_coding_rna):.4f}</li>
                <li>编码RNA: {float(prob_coding_rna):.4f}</li>
            </ul>
            <p>原始输出值: {np.array2string(raw_model_output, formatter={'float_kind': lambda x: "%.4f" % x})}</p>
            """

        self.result_label.setText(result_html_text)
        # self.predict_btn.setEnabled(True) # 移至 on_worker_finished

    def handle_prediction_error(self, error_message):
        """处理预测过程中发生的错误。"""
        self.progress_bar.setValue(0)
        # 尝试将进度条变为红色或显示错误状态 - 保持原始进度条样式，仅更新文本
        # 原始GUI中没有改变进度条颜色，所以这里也不改变
        self.status_label.setText("发生错误")  # 与原始GUI一致
        self.show_error_message(f"预测过程中出现错误:\n{error_message}")  # 与原始GUI一致
        # self.predict_btn.setEnabled(True) # 移至 on_worker_finished

    def on_worker_finished(self):
        """当 PredictionWorker 线程完成时调用 (无论成功或失败)。"""
        self.predict_btn.setEnabled(True)
        # 如果没有错误，进度条应为100%，否则为0%
        if "错误" not in self.status_label.text() and "Error" not in self.status_label.text():
            if self.progress_bar.value() < 100:  # 如果因为某些原因没有达到100
                self.progress_bar.setValue(100)  # 对于成功完成的设置为100
        else:
            self.progress_bar.setValue(0)  # 错误时重置为0

        if "完成" not in self.status_label.text().lower() and \
                "错误" not in self.status_label.text().lower() and \
                "error" not in self.status_label.text().lower():
            self.status_label.setText("准备进行下一次预测。")

        if self.prediction_temp_dir_manager:
            try:
                self.prediction_temp_dir_manager.cleanup()
                print(f"临时目录 {self.prediction_temp_dir_manager.name} 已清理。")
            except Exception as e:
                print(f"清理临时目录时出错: {e}")
            finally:
                self.prediction_temp_dir_manager = None

        # 确保进度条样式恢复 (如果曾改变过) - 此处不需要，因为我们不改变错误时的颜色

    def show_error_message(self, message):
        """在 QMessageBox 中显示错误消息 - 与原始 GUI.txt 一致。"""
        QMessageBox.critical(self, "错误", message)  # "错误" 为窗口标题
        # self.predict_btn.setEnabled(True) # 已移至 on_worker_finished 或特定错误处理点

    def closeEvent(self, event):
        """确保在应用程序关闭时清理临时目录。"""
        if self.prediction_temp_dir_manager:
            try:
                self.prediction_temp_dir_manager.cleanup()
                print("应用程序关闭，临时目录已清理。")
            except Exception as e:
                print(f"关闭应用程序时清理临时目录出错: {e}")
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # 设置高DPI支持
    if hasattr(Qt.ApplicationAttribute, 'AA_EnableHighDpiScaling'):
        app.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    if hasattr(Qt.ApplicationAttribute, 'AA_UseHighDpiPixmaps'):
        app.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)

    window = ModernWindow()
    window.show()
    sys.exit(app.exec())
