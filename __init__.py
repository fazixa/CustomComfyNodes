from .video_utils import VideoChangeFramerateNode
from .pink_extractor import PinkExtractorNode, ColorExtractorNode, PinkOutlineZonesNode, MaskOuterRingNode, MaskCenteredStrokeNode, MaskStrokeNode, MaskDilateColorNode
from .blender_gp_trace import BlenderGPTraceNode
from .iphone_frame import iPhoneVideoFrameNode, iPhoneHLGCorrectNode
from .pipo_align_composite import PipoAlignCompositeNode
from .pipo_align_restore import PipoAlignRestoreNode
from .boil_effect import BoilEffectNode
from .seedance import SeedanceGenerateNode
from .outline_roughen import OutlineRoughenNode
from .mask_fill_gaps import MaskFillGapsNode
from .mask_track import MaskTrackNode
from .sam2_node import SAM2SegmentNode, SAM2SegmentVideoNode
from . import sam2_routes  # noqa: F401 — registers /fae/sam2/* on import

NODE_CLASS_MAPPINGS = {
    "VideoChangeFramerate": VideoChangeFramerateNode,
    "PinkExtractor": PinkExtractorNode,
    "ColorExtractor": ColorExtractorNode,
    "PinkOutlineZones": PinkOutlineZonesNode,
    "MaskOuterRing": MaskOuterRingNode,
    "MaskCenteredStroke": MaskCenteredStrokeNode,
    "MaskStroke": MaskStrokeNode,
    "MaskDilateColor": MaskDilateColorNode,
    "BlenderGPTrace": BlenderGPTraceNode,
    "iPhoneVideoFrame": iPhoneVideoFrameNode,
    "iPhoneHLGCorrect": iPhoneHLGCorrectNode,
    "PipoAlignComposite": PipoAlignCompositeNode,
    "PipoAlignRestore": PipoAlignRestoreNode,
    "BoilEffect": BoilEffectNode,
    "SeedanceGenerate": SeedanceGenerateNode,
    "OutlineRoughen": OutlineRoughenNode,
    "MaskFillGaps": MaskFillGapsNode,
    "MaskTrack": MaskTrackNode,
    "SAM2Segment": SAM2SegmentNode,
    "SAM2SegmentVideo": SAM2SegmentVideoNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoChangeFramerate": "Video Change Framerate",
    "PinkExtractor": "Pink Extractor",
    "ColorExtractor": "Color Extractor",
    "PinkOutlineZones": "Pink Outline Zones",
    "MaskOuterRing": "Mask Outer Ring",
    "MaskCenteredStroke": "Mask Centered Stroke",
    "MaskStroke": "Mask Stroke",
    "MaskDilateColor": "Mask Dilate Color",
    "BlenderGPTrace": "Blender GP Trace",
    "iPhoneVideoFrame": "iPhone Video Frame",
    "iPhoneHLGCorrect": "iPhone HLG Correct",
    "PipoAlignComposite": "Pipo Align & Composite",
    "PipoAlignRestore": "Pipo Align & Restore",
    "BoilEffect": "Boil Effect",
    "SeedanceGenerate": "Seedance 2.0 Generate",
    "OutlineRoughen": "Outline Roughen",
    "MaskFillGaps": "Mask Fill Gaps",
    "MaskTrack": "Mask Track",
    "SAM2Segment": "SAM2 Segment",
    "SAM2SegmentVideo": "SAM2 Segment Video",
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
