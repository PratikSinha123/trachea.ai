"""CT scan preprocessing utilities — normalization, resampling, and windowing."""

import numpy as np
import SimpleITK as sitk


# Standard Hounsfield Unit ranges for airway analysis
HU_AIR_MIN = -1024
HU_AIR_MAX = -200
HU_TISSUE_MIN = -500
HU_TISSUE_MAX = 400
HU_WINDOW_MIN = -1024
HU_WINDOW_MAX = 600


def normalize_hu(image: sitk.Image, window_min=HU_WINDOW_MIN,
                 window_max=HU_WINDOW_MAX) -> sitk.Image:
    """Normalize CT image Hounsfield Units to [0, 1] range.

    Clamps values to the specified window and scales linearly.
    """
    clamped = sitk.Clamp(image, sitk.sitkFloat32, window_min, window_max)
    shifted = sitk.ShiftScale(clamped, -window_min, 1.0 / (window_max - window_min))
    return shifted


def resample_isotropic(image: sitk.Image, new_spacing=(1.0, 1.0, 1.0),
                       interpolator=sitk.sitkLinear) -> sitk.Image:
    """Resample image to isotropic voxel spacing.

    This standardizes the resolution across different scanners and protocols,
    which is critical for consistent segmentation quality.
    """
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    new_size = [
        int(round(osz * ospc / nspc))
        for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(-1024)

    return resampler.Execute(image)


def resample_mask_like(mask: sitk.Image, reference: sitk.Image) -> sitk.Image:
    """Resample a binary mask to match a reference image's geometry."""
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(mask)


def extract_lung_region(image: sitk.Image) -> sitk.Image:
    """Extract a rough lung region to narrow down the search area for trachea.

    Uses thresholding on the air-filled regions and keeps the largest
    connected components (the two lungs + trachea).
    """
    arr = sitk.GetArrayFromImage(image)

    # Air-filled regions (HU < -200)
    air_mask = (arr < HU_AIR_MAX).astype(np.uint8)

    air_sitk = sitk.GetImageFromArray(air_mask)
    air_sitk.CopyInformation(image)

    # Remove small objects, keep large connected air regions
    cc = sitk.ConnectedComponent(air_sitk)
    relabeled = sitk.RelabelComponent(cc, minimumObjectSize=5000)

    # Keep the top components (lungs + trachea are the largest air regions)
    binary = sitk.BinaryThreshold(relabeled, 1, 100)
    return binary


def crop_to_roi(image: sitk.Image, mask: sitk.Image, padding=10):
    """Crop image and mask to the bounding box of the mask with padding.

    Reduces memory usage and speeds up subsequent processing.
    """
    label_shape = sitk.LabelShapeStatisticsImageFilter()
    label_shape.Execute(mask)

    if label_shape.GetNumberOfLabels() == 0:
        return image, mask

    bbox = label_shape.GetBoundingBox(1)
    ndim = image.GetDimension()

    start = list(bbox[:ndim])
    size = list(bbox[ndim:])

    image_size = image.GetSize()
    for i in range(ndim):
        start[i] = max(0, start[i] - padding)
        end_i = min(image_size[i], start[i] + size[i] + 2 * padding)
        size[i] = end_i - start[i]

    cropped_image = sitk.RegionOfInterest(image, size, start)
    cropped_mask = sitk.RegionOfInterest(mask, size, start)

    return cropped_image, cropped_mask


def preprocess_ct(image: sitk.Image, target_spacing=(0.75, 0.75, 0.75)):
    """Full preprocessing pipeline for a CT image.

    Steps:
        1. Resample to isotropic spacing
        2. Extract lung region
        3. Normalize HU values

    Returns:
        tuple: (resampled_image, lung_mask, normalized_image)
    """
    print(f"  Original spacing: {image.GetSpacing()}, size: {image.GetSize()}")

    resampled = resample_isotropic(image, new_spacing=target_spacing)
    print(f"  Resampled spacing: {resampled.GetSpacing()}, size: {resampled.GetSize()}")

    lung_mask = extract_lung_region(resampled)
    normalized = normalize_hu(resampled)

    return resampled, lung_mask, normalized


def prepare_for_unet(image: sitk.Image, patch_size=(64, 128, 128)):
    """Prepare a CT volume for 3D U-Net inference by extracting overlapping patches.

    Returns:
        list: List of (patch_array, origin_index) tuples
    """
    arr = sitk.GetArrayFromImage(image)

    # Normalize to [0, 1]
    arr = arr.astype(np.float32)
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max > arr_min:
        arr = (arr - arr_min) / (arr_max - arr_min)

    pz, py, px = patch_size
    sz, sy, sx = arr.shape

    # Calculate strides with 50% overlap
    stride_z = max(1, pz // 2)
    stride_y = max(1, py // 2)
    stride_x = max(1, px // 2)

    patches = []
    for z in range(0, max(1, sz - pz + 1), stride_z):
        for y in range(0, max(1, sy - py + 1), stride_y):
            for x in range(0, max(1, sx - px + 1), stride_x):
                z_end = min(z + pz, sz)
                y_end = min(y + py, sy)
                x_end = min(x + px, sx)

                z_start = z_end - pz
                y_start = y_end - py
                x_start = x_end - px

                patch = arr[z_start:z_end, y_start:y_end, x_start:x_end]
                patches.append((patch, (z_start, y_start, x_start)))

    # Handle case where volume is smaller than patch size
    if not patches:
        padded = np.zeros(patch_size, dtype=np.float32)
        padded[:min(sz, pz), :min(sy, py), :min(sx, px)] = arr[
            :min(sz, pz), :min(sy, py), :min(sx, px)
        ]
        patches.append((padded, (0, 0, 0)))

    return patches


def reconstruct_from_patches(patches, output_shape, patch_size=(64, 128, 128)):
    """Reconstruct a full volume from overlapping patches by averaging.

    Args:
        patches: list of (prediction_array, origin_index) tuples
        output_shape: shape of the full output volume (Z, Y, X)
        patch_size: size of each patch

    Returns:
        numpy.ndarray: Reconstructed volume
    """
    output = np.zeros(output_shape, dtype=np.float32)
    counts = np.zeros(output_shape, dtype=np.float32)

    pz, py, px = patch_size

    for pred, (z, y, x) in patches:
        output[z:z + pz, y:y + py, x:x + px] += pred
        counts[z:z + pz, y:y + py, x:x + px] += 1.0

    counts = np.maximum(counts, 1.0)
    return output / counts
