from .video_utils import VideoChangeFramerateNode
from .pink_extractor import PinkExtractorNode
from .blender_gp_trace import BlenderGPTraceNode
from .iphone_frame import iPhoneVideoFrameNode, iPhoneHLGCorrectNode
from .pipo_align_composite import PipoAlignCompositeNode
from .boil_effect import BoilEffectNode

NODE_CLASS_MAPPINGS = {
    "VideoChangeFramerate": VideoChangeFramerateNode,
    "PinkExtractor": PinkExtractorNode,
    "BlenderGPTrace": BlenderGPTraceNode,
    "iPhoneVideoFrame": iPhoneVideoFrameNode,
    "iPhoneHLGCorrect": iPhoneHLGCorrectNode,
    "PipoAlignComposite": PipoAlignCompositeNode,
    "BoilEffect": BoilEffectNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoChangeFramerate": "Video Change Framerate",
    "PinkExtractor": "Pink Extractor",
    "BlenderGPTrace": "Blender GP Trace",
    "iPhoneVideoFrame": "iPhone Video Frame",
    "iPhoneHLGCorrect": "iPhone HLG Correct",
    "PipoAlignComposite": "Pipo Align & Composite",
    "BoilEffect": "Boil Effect",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
