"""
Document Loader Module
Supports loading PDF, PPT, DOCX, and Image files
"""

from abc import ABC, abstractmethod     #抽象基类
from pathlib import Path                #处理文件路径
from typing import List, Optional       #类型提示
from dataclasses import dataclass       #数据类，装饰器使用

#Document类
#读取后的文档格式
@dataclass
class Document:
    """Represents a loaded document with metadata"""
    content: str           #文档里面的文字内容
    file_path: str         #文件路径
    file_type: str         #文件类型（pdf/pot/docx/image）
    metadata: dict         #额外信息（页数，幻灯片数..）

#BaseLoader 抽象类（模板）
class BaseLoader(ABC):
    """Base class for document loaders"""

    #读取文件
    @abstractmethod
    def load(self, file_path: str) -> Document:
        """Load a document from the given file path"""
        pass
    #判断是否支持该类型
    @abstractmethod
    def supports(self, file_extension: str) -> bool:
        """Check if this loader supports the given file extension"""
        pass

#PDFLoader读取PDF
class PDFLoader(BaseLoader):
    """Loader for PDF files"""

    #接收一个文件路径，返回一个Document对象
    def load(self, file_path: str) -> Document:
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError("Please install pypdf: pip install pypdf")
        #创建一个PDF阅读器对象
        reader = PdfReader(file_path)
        content = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                content.append(text)

        return Document(
            content="\n".join(content),   #将每页的文本片段连接成一段长文本，使用\n隔离
            file_path=file_path,
            file_type="pdf",
            metadata={"num_pages": len(reader.pages)}
        )

    def supports(self, file_extension: str) -> bool:
        return file_extension.lower() == ".pdf"

#读取ppt
class PPTLoader(BaseLoader):
    """Loader for PPT/PPTX files"""   #加载ppt文件

    def load(self, file_path: str) -> Document:
        try:
            from pptx import Presentation
        except ImportError:
            raise ImportError("Please install python-pptx: pip install python-pptx")

        #创建ppt加载起对象
        prs = Presentation(file_path)
        content = []
        #prs.slides 幻灯片数  i索引 slide幻灯片对象
        for i, slide in enumerate(prs.slides):
            slide_text = []
            #遍历当前幻灯片的所有元素
            for shape in slide.shapes:
                #检查幻灯片是否有“text”元素，并且text元素是否不为空
                if hasattr(shape, "text") and shape.text:
                    slide_text.append(shape.text)
            if slide_text:
                #提取当前幻灯片的text的内容，[Slide 页数{i+1}]\n加上一个幻灯片的
                content.append(f"[Slide {i+1}]\n" + "\n".join(slide_text))

        return Document(
            content="\n".join(content),
            file_path=file_path,
            file_type="ppt",
            metadata={"num_slides": len(prs.slides)}
        )

    def supports(self, file_extension: str) -> bool:
        return file_extension.lower() in [".ppt", ".pptx"]

#读取docx
class DOCXLoader(BaseLoader):
    """Loader for DOCX files"""

    def load(self, file_path: str) -> Document:
        try:
            from docx import Document as DocxDocument
        except ImportError:
            raise ImportError("Please install python-docx: pip install python-docx")

        doc = DocxDocument(file_path)
        #将word文档的非空段落提取出来，存成一个干净的文本列表
        paragraphs = [para.text for para in doc.paragraphs if para.text]

        return Document(
            content="\n".join(paragraphs),
            file_path=file_path,
            file_type="docx",
            metadata={"num_paragraphs": len(paragraphs)}
        )

    def supports(self, file_extension: str) -> bool:
        return file_extension.lower() == ".docx"

#图片加载器 核心能力是OCR（光学字符识别）
class ImageLoader(BaseLoader):
    """Loader for Image files with OCR support"""
    #参数lang  chi_simt：同时识别中文和英文
    def __init__(self, lang: str = "chi_sim+eng"):
        self.lang = lang

    def load(self, file_path: str) -> Document:
        try:
            import pytesseract            #将图片转成文字
            from PIL import Image         #图像处理库，打开图片文件，将图片数据交给Tesseract
        except ImportError:
            raise ImportError("Please install pytesseract and Pillow: pip install pytesseract Pillow")
        #创建一个Image对象
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image, lang=self.lang)        #image传入图片，  设置中英文  转换成字符串存入text

        return Document(
            content=text,
            file_path=file_path,
            file_type="image",
            metadata={"image_size": image.size, "mode": image.mode}     #图片大小，图片的模式（RGB彩图还是灰度图）
        )

    def supports(self, file_extension: str) -> bool:
        return file_extension.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".gif"]


class DocumentLoader:
    """
    Unified document loader that automatically selects the appropriate loader
    based on file extension
    """
    #在构造构造函数初始化这些工具类    ->设计模式  策略模式类似于工厂模式 好处：一次配置，终身受用
    def __init__(self):
        self.loaders: List[BaseLoader] = [
            PDFLoader(),
            PPTLoader(),
            DOCXLoader(),
            ImageLoader()
        ]
    #传入文件路径，判断是用哪个文件解析器
    def load(self, file_path: str) -> Document:
        """Load a document from the given file path"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        #获取文件后缀名
        extension = path.suffix
        for loader in self.loaders:
            #分发，看它支持哪个文件格式
            if loader.supports(extension):
                return loader.load(file_path)

        raise ValueError(f"Unsupported file type: {extension}")
        #传入目录路径
    def load_directory(self, directory_path: str, file_types: Optional[List[str]] = None) -> List[Document]:
        """Load all documents from a directory"""
        path = Path(directory_path)
        if not path.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory_path}")

        documents = []
        #path.iterdir  这是一个生成器，他会一个一个列出文件夹的所有内容（包括子文件夹和文件）
        for file_path in path.iterdir():
            if file_types is None or file_path.suffix in file_types:   #如果没在指定类型全部接收，如果指定了就选择指定的文件
                try:
                    #传给文档解析器
                    doc = self.load(str(file_path))
                    #将文档解析起解析的内容 追加document
                    documents.append(doc)
                except (ValueError, ImportError) as e:
                    print(f"Skipping {file_path}: {e}")

        return documents


if __name__ == "__main__":
    # Example usage
    loader = DocumentLoader()
    doc = loader.load("example.pdf")
    print(f"Loaded {doc.file_type} with {len(doc.content)} characters")
