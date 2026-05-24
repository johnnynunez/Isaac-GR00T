# --------------------------------------------------------
# NVIDIA
# Copyright (c) 2025 NVIDIA
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

# copy from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava_onevision/image_processing_llava_onevision_fast.py
from typing import List, Optional, Union

from transformers.image_processing_utils import (
    BatchFeature,
    get_patch_output_size,
    select_best_resolution,
)
from transformers.image_processing_utils_fast import (
    BaseImageProcessorFast,
    DefaultFastImageProcessorKwargs,
    divide_to_patches,
    group_images_by_shape,
    reorder_images,
)
# GR00T16_RLINF_COMPAT: transformers 4.55+ dropped the explicit
# ``BASE_IMAGE_PROCESSOR_FAST_DOCSTRING*`` constants and switched to
# the ``@auto_docstring`` decorator. The constants are only consumed
# by ``add_start_docstrings`` for human-readable docs, so empty strings
# preserve runtime semantics.
try:
    from transformers.image_processing_utils_fast import (  # type: ignore
        BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,
        BASE_IMAGE_PROCESSOR_FAST_DOCSTRING_PREPROCESS,
    )
except ImportError:  # pragma: no cover - transformers >= 4.55
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING = ""
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING_PREPROCESS = ""
from transformers.image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    IMAGENET_STANDARD_MEAN,  # 0.5, 0.5, 0.5
    IMAGENET_STANDARD_STD,  # 0.5, 0.5, 0.5
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    SizeDict,
    get_image_size,
    make_flat_list_of_images,
    validate_kwargs,
)
# GR00T16_RLINF_COMPAT: ``VideoInput`` and ``make_batched_videos`` moved
# from ``transformers.image_utils`` to the new ``transformers.video_utils``
# in 4.55+. Keep importing them under the same names regardless of which
# release is installed.
try:
    from transformers.video_utils import (  # type: ignore
        VideoInput,
        make_batched_videos,
    )
except ImportError:  # pragma: no cover - transformers < 4.55
    from transformers.image_utils import (  # noqa: F401
        VideoInput,
        make_batched_videos,
    )
from transformers.processing_utils import Unpack
from transformers.utils import (
    TensorType,
    add_start_docstrings,
    is_torch_available,
    is_torchvision_v2_available,
)


if is_torch_available():
    import torch
if is_torchvision_v2_available():
    from transformers.image_utils import pil_torch_interpolation_mapping

    from torchvision.transforms.v2 import functional as F
else:
    from torchvision.transforms import functional as F


def crop(
    img: torch.Tensor, left: int, top: int, right: int, bottom: int
) -> torch.Tensor:
    """Crop the given numpy array.

    Args:
        img (torch.Tensor): Image to be cropped. Format should be (C, H, W).
        left (int): The left coordinate of the crop box.
        top (int): The top coordinate of the crop box.
        right (int): The right coordinate of the crop box.
        bottom (int): The bottom coordinate of the crop box.

    Returns:
        torch.Tensor: Cropped image.
    """
    if not isinstance(img, torch.Tensor):
        raise TypeError("img should be torch.Tensor. Got {}".format(type(img)))

    if img.ndim not in [2, 3]:
        raise ValueError("Image should have 2 or 3 dimensions. Got {}".format(img.ndim))

    img_height = img.shape[1]
    img_width = img.shape[2]
    if top < 0 or left < 0 or bottom > img_height or right > img_width:
        raise ValueError("Crop coordinates out of bounds")

    if top >= bottom or left >= right:
        raise ValueError("Invalid crop coordinates")

    return img[:, top:bottom, left:right]


class Eagle3_VLFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    do_pad: Optional[bool]


@add_start_docstrings(
    "Constructs a fast ConvNeXT image processor. Based on [`SiglipImageProcessor`] with incorporation of processing each video frame.",
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,
    """
        image_grid_pinpoints (`List[List[int]]`, *optional*):
            A list of possible resolutions to use for processing high resolution images. The best resolution is selected
            based on the original size of the image. Can be overridden by `image_grid_pinpoints` in the `preprocess`
            method. Not used for processing videos.
        do_pad (`bool`, *optional*):
            Whether to pad the image. If `True`, will pad the patch dimension of the images in the batch to the largest
            number of patches in the batch. Padding will be applied to the bottom and right with zeros.
    """,
)
class Eagle3_VLImageProcessorFast(BaseImageProcessorFast):
    resample = PILImageResampling.BICUBIC
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    size = {"height": 448, "width": 448}
    default_to_square = False
    crop_size = None
    do_resize = True
    do_center_crop = None
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    do_pad = True
    valid_kwargs = Eagle3_VLFastImageProcessorKwargs
    model_input_names = ["pixel_values_videos"]

    def __init__(self, **kwargs: Unpack[Eagle3_VLFastImageProcessorKwargs]):
        super().__init__(**kwargs)

    @add_start_docstrings(
        BASE_IMAGE_PROCESSOR_FAST_DOCSTRING_PREPROCESS,
        """
            do_pad (`bool`, *optional*):
                    Whether to pad the image. If `True`, will pad the patch dimension of the images in the batch to the largest
                    number of patches in the batch. Padding will be applied to the bottom and right with zeros.
        """,
    )
    def preprocess(
        self, images: ImageInput, **kwargs: Unpack[Eagle3_VLFastImageProcessorKwargs]
    ) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def _prepare_images_structure(
        self,
        images: ImageInput,
        expected_ndims: int = 3,
    ) -> ImageInput:
        """
        Prepare the images structure for processing.

        Args:
            images (`ImageInput`):
                The input images to process.

        Returns:
            `ImageInput`: The images with a valid nesting.
        """
        # GR00T16_RLINF_COMPAT: transformers 4.55+ calls processor overrides
        # through BaseImageProcessorFast._prepare_image_like_inputs(...), which
        # forwards expected_ndims into _prepare_images_structure. Eagle3's
        # original implementation ignored that parameter, so accept it for
        # compatibility while preserving Eagle3's flattening behavior.
        del expected_ndims
        return make_flat_list_of_images(images)

    def _preprocess(
        self,
        images: List["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        interpolation: Optional["F.InterpolationMode"],
        do_center_crop: bool,
        crop_size: SizeDict,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Optional[Union[float, List[float]]],
        image_std: Optional[Union[float, List[float]]],
        do_pad: bool,
        return_tensors: Optional[Union[str, TensorType]],
        disable_grouping: bool = False,
    ) -> BatchFeature:

        image_sizes = [
            get_image_size(image, channel_dim=ChannelDimension.FIRST)
            for image in images
        ]

        # Group images by size for further processing
        # Needed in case do_resize is False, or resize returns images with different sizes
        grouped_images, grouped_images_index = group_images_by_shape(
            images, disable_grouping=disable_grouping
        )
        processed_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            # Fused rescale and normalize
            stacked_images = self.rescale_and_normalize(
                stacked_images,
                do_rescale,
                rescale_factor,
                do_normalize,
                image_mean,
                image_std,
            )
            processed_images_grouped[shape] = stacked_images

        processed_images = reorder_images(
            processed_images_grouped, grouped_images_index
        )
        processed_images = torch.stack(processed_images)

        return BatchFeature(
            data={"pixel_values": processed_images, "image_sizes": image_sizes},
            tensor_type=return_tensors,
        )

    def preprocess(
        self,
        images: ImageInput,
        videos: VideoInput = None,
        **kwargs: Unpack[Eagle3_VLFastImageProcessorKwargs],
    ) -> BatchFeature:
        # GR00T16_RLINF_COMPAT: on newer transformers/Python, the subclass
        # TypedDict annotations exposed here may contain only Eagle3 additions
        # (currently do_pad) rather than inherited DefaultFastImageProcessorKwargs
        # keys. Merge both sets so defaults for resample, size, data_format, etc.
        # are populated before the legacy Eagle3 code pops them below.
        valid_kwarg_names = {
            **DefaultFastImageProcessorKwargs.__annotations__,
            **self.valid_kwargs.__annotations__,
        }
        validate_kwargs(
            captured_kwargs=kwargs.keys(),
            valid_processor_keys=valid_kwarg_names.keys(),
        )
        # Set default kwargs from self. This ensures that if a kwarg is not provided
        # by the user, it gets its default value from the instance, or is set to None.
        for kwarg_name in valid_kwarg_names:
            kwargs.setdefault(kwarg_name, getattr(self, kwarg_name, None))

        # Extract parameters that are only used for preparing the input images
        do_convert_rgb = kwargs.pop("do_convert_rgb")
        input_data_format = kwargs.pop("input_data_format")
        device = kwargs.pop("device")
        # GR00T16_RLINF_COMPAT: transformers 4.55+ renamed the helper
        # ``BaseImageProcessorFast._prepare_input_images`` to
        # ``_prepare_image_like_inputs``. Pick whichever exists so the
        # processor works on both old (4.51.x) and new (4.55+) installs.
        _prep_imgs = getattr(self, "_prepare_input_images", None) or getattr(
            self, "_prepare_image_like_inputs", None
        )
        if _prep_imgs is None:
            raise RuntimeError(
                "BaseImageProcessorFast has neither _prepare_input_images "
                "nor _prepare_image_like_inputs; transformers version not supported."
            )
        # Prepare input images
        if images is not None:
            images = _prep_imgs(
                images=images,
                do_convert_rgb=do_convert_rgb,
                input_data_format=input_data_format,
                device=device,
            )

        if videos is not None:
            videos = _prep_imgs(
                images=videos,
                do_convert_rgb=do_convert_rgb,
                input_data_format=input_data_format,
                device=device,
            )

        # Update kwargs that need further processing before being validated
        kwargs = self._further_process_kwargs(**kwargs)

        # Validate kwargs
        self._validate_preprocess_kwargs(**kwargs)

        # GR00T16_RLINF_COMPAT: transformers 4.55+ already pops ``resample``
        # and emits ``interpolation`` inside BaseImageProcessorFast.
        # _further_process_kwargs(). Older transformers leave ``resample`` for
        # this legacy Eagle3 block. Support both without double-popping.
        if "interpolation" not in kwargs:
            resample = kwargs.pop("resample")
            kwargs["interpolation"] = (
                pil_torch_interpolation_mapping[resample]
                if isinstance(resample, (PILImageResampling, int))
                else resample
            )
        else:
            kwargs.pop("resample", None)

        # Pop kwargs that are not accepted by Eagle3's legacy _preprocess
        # signature. Newer transformers DefaultFastImageProcessorKwargs adds
        # fields such as disable_grouping/pad_size for the generic base class;
        # forwarding them here raises TypeError.
        for unused_kwarg in (
            "default_to_square",
            "data_format",
            "pad_size",
        ):
            kwargs.pop(unused_kwarg, None)
        if images is not None:
            return self._preprocess(images, **kwargs)
        elif videos is not None:
            return self._preprocess(videos, **kwargs)


__all__ = ["Eagle3_VLImageProcessorFast"]
