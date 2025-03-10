from __future__ import annotations

from functools import partial
from itertools import count
from typing import Any, Literal, overload

from vsexprtools import ExprOp, ExprVars, complexpr_available, norm_expr
from vskernels import Bilinear, Gaussian
from vstools import (
    ConvMode, CustomValueError, FunctionUtil, OneDimConvModeT, PlanesT, SpatialConvModeT, KwargsNotNone,
    TempConvModeT, check_variable, core, join, normalize_planes, normalize_seq, split, to_arr, vs
)

from .enum import BlurMatrix, BlurMatrixBase, LimitFilterMode, BilateralBackend
from .freqs import MeanMode
from .limit import limit_filter
from .util import normalize_radius

__all__ = [
    'box_blur', 'side_box_blur',
    'gauss_blur',
    'min_blur', 'sbr', 'median_blur',
    'bilateral', 'flux_smooth'
]


def box_blur(
    clip: vs.VideoNode, radius: int | list[int] = 1, passes: int = 1,
    mode: OneDimConvModeT | TempConvModeT = ConvMode.HV, planes: PlanesT = None, **kwargs: Any
) -> vs.VideoNode:
    assert check_variable(clip, box_blur)

    planes = normalize_planes(clip, planes)

    if isinstance(radius, list):
        return normalize_radius(clip, box_blur, radius, planes, passes=passes)

    if not radius:
        return clip

    if mode == ConvMode.TEMPORAL:
        return BlurMatrix.MEAN(radius, mode=mode)(clip, planes, passes=passes, **kwargs)

    box_args = (
        planes,
        radius, 0 if mode == ConvMode.VERTICAL else passes,
        radius, 0 if mode == ConvMode.HORIZONTAL else passes
    )

    return clip.vszip.BoxBlur(*box_args)


def side_box_blur(
    clip: vs.VideoNode, radius: int | list[int] = 1, planes: PlanesT = None,
    inverse: bool = False
) -> vs.VideoNode:
    planes = normalize_planes(clip, planes)

    if isinstance(radius, list):
        return normalize_radius(clip, side_box_blur, radius, planes, inverse=inverse)

    half_kernel = [(1 if i <= 0 else 0) for i in range(-radius, radius + 1)]

    conv_m1 = partial(core.std.Convolution, matrix=half_kernel, planes=planes)
    conv_m2 = partial(core.std.Convolution, matrix=half_kernel[::-1], planes=planes)
    blur_pt = partial(box_blur, planes=planes)

    vrt_filters, hrz_filters = [
        [
            partial(conv_m1, mode=mode), partial(conv_m2, mode=mode),
            partial(blur_pt, hradius=hr, vradius=vr, hpasses=h, vpasses=v)
        ] for h, hr, v, vr, mode in [
            (0, None, 1, radius, ConvMode.VERTICAL), (1, radius, 0, None, ConvMode.HORIZONTAL)
        ]
    ]

    vrt_intermediates = (vrt_flt(clip) for vrt_flt in vrt_filters)
    intermediates = list(
        hrz_flt(vrt_intermediate)
        for i, vrt_intermediate in enumerate(vrt_intermediates)
        for j, hrz_flt in enumerate(hrz_filters) if not i == j == 2
    )

    comp_blur = None if inverse else box_blur(clip, radius, 1, planes=planes)

    if complexpr_available:
        template = '{cum} x - abs {new} x - abs < {cum} {new} ?'

        cum_expr, cumc = '', 'y'
        n_inter = len(intermediates)

        for i, newc, var in zip(count(), ExprVars[2:26], ExprVars[4:26]):
            if i == n_inter - 1:
                break

            cum_expr += template.format(cum=cumc, new=newc)

            if i != n_inter - 2:
                cumc = var.upper()
                cum_expr += f' {cumc}! '
                cumc = f'{cumc}@'

        if comp_blur:
            clips = [clip, *intermediates, comp_blur]
            cum_expr = f'x {cum_expr} - {ExprVars[n_inter + 1]} +'
        else:
            clips = [clip, *intermediates]

        cum = norm_expr(clips, cum_expr, planes, func=side_box_blur)
    else:
        cum = intermediates[0]
        for new in intermediates[1:]:
            cum = limit_filter(clip, cum, new, LimitFilterMode.SIMPLE2_MIN, planes)

        if comp_blur:
            cum = clip.std.MakeDiff(cum).std.MergeDiff(comp_blur)

    if comp_blur:
        return box_blur(cum, 1, min(radius // 2, 1))

    return cum


def gauss_blur(
    clip: vs.VideoNode, sigma: float | list[float] = 0.5, taps: int | None = None,
    mode: ConvMode = ConvMode.HV, planes: PlanesT = None,
    **kwargs: Any
) -> vs.VideoNode:
    assert check_variable(clip, gauss_blur)

    planes = normalize_planes(clip, planes)

    if isinstance(sigma, list):
        return normalize_radius(clip, gauss_blur, ('sigma', sigma), planes, mode=mode)

    fast = kwargs.pop("_fast", False)

    sigma_constant = 0.9 if fast and not mode.is_temporal else sigma
    taps = BlurMatrix.GAUSS.get_taps(sigma_constant, taps)

    if not mode.is_temporal:
        def _resize2_blur(plane: vs.VideoNode, sigma: float, taps: int) -> vs.VideoNode:
            resize_kwargs = dict[str, Any]()

            # Downscale approximation can be used by specifying _fast=True
            # Has a big speed gain when taps is large
            if fast:
                wdown, hdown = plane.width, plane.height

                if ConvMode.VERTICAL in mode:
                    hdown = round(max(round(hdown / sigma), 2) / 2) * 2

                if ConvMode.HORIZONTAL in mode:
                    wdown = round(max(round(wdown / sigma), 2) / 2) * 2

                resize_kwargs.update(width=plane.width, height=plane.height)

                plane = Bilinear.scale(plane, wdown, hdown)
                sigma = sigma_constant
            else:
                resize_kwargs.update({f'force_{k}': k in mode for k in 'hv'})

            return Gaussian(sigma, taps).scale(plane, **resize_kwargs | kwargs)

        if not {*range(clip.format.num_planes)} - {*planes}:
            return _resize2_blur(clip, sigma, taps)

        return join([
            _resize2_blur(p, sigma, taps) if i in planes else p
            for i, p in enumerate(split(clip))
        ])

    kernel: BlurMatrixBase[float] = BlurMatrix.GAUSS(  # type: ignore
        taps, sigma=sigma, mode=mode, scale_value=1023
    )

    return kernel(clip, planes, **kwargs)


def min_blur(
    clip: vs.VideoNode, radius: int | list[int] = 1,
    mode: tuple[ConvMode, ConvMode] = (ConvMode.HV, ConvMode.SQUARE), planes: PlanesT = None,
    **kwargs: Any
) -> vs.VideoNode:
    """
    MinBlur by Didée (http://avisynth.nl/index.php/MinBlur)
    Nifty Gauss/Median combination
    """
    assert check_variable(clip, min_blur)

    planes = normalize_planes(clip, planes)

    if isinstance(radius, list):
        return normalize_radius(clip, min_blur, radius, planes)

    mode_blur, mode_median = normalize_seq(mode, 2)

    blurred = BlurMatrix.BINOMIAL(radius=radius, mode=mode_blur)(clip, planes=planes, **kwargs)
    median = median_blur(clip, radius, mode_median, planes=planes)

    return MeanMode.MEDIAN([clip, blurred, median], planes=planes)


def sbr(
    clip: vs.VideoNode, radius: int | list[int] = 1,
    mode: ConvMode = ConvMode.HV, planes: PlanesT = None,
    **kwargs: Any
) -> vs.VideoNode:
    assert check_variable(clip, sbr)

    planes = normalize_planes(clip, planes)

    blur_kernel = BlurMatrix.BINOMIAL(radius=radius, mode=mode)

    blurred = blur_kernel(clip, planes=planes, **kwargs)

    diff = clip.std.MakeDiff(blurred, planes=planes)
    blurred_diff = blur_kernel(diff, planes=planes, **kwargs)

    return norm_expr(
        [clip, diff, blurred_diff],
        'y z - D1! y neutral - D2! x D1@ D2@ xor 0 D1@ abs D2@ abs < D1@ D2@ ? ? -',
        planes=planes, func=sbr
    )


@overload
def median_blur(
    clip: vs.VideoNode, radius: int = ..., mode: Literal[ConvMode.TEMPORAL] = ..., planes: PlanesT = ...
) -> vs.VideoNode:
    ...


@overload
def median_blur(
    clip: vs.VideoNode, radius: int | list[int] = ..., mode: SpatialConvModeT = ..., planes: PlanesT = None
) -> vs.VideoNode:
    ...


@overload
def median_blur(
    clip: vs.VideoNode, radius: int | list[int] = ..., mode: ConvMode = ..., planes: PlanesT = None
) -> vs.VideoNode:
    ...


def median_blur(
    clip: vs.VideoNode, radius: int | list[int] = 1, mode: ConvMode = ConvMode.SQUARE, planes: PlanesT = None
) -> vs.VideoNode:
    if mode == ConvMode.TEMPORAL:
        if isinstance(radius, int):
            return clip.zsmooth.TemporalMedian(radius, planes)

        raise CustomValueError("A list of radius isn't supported for ConvMode.TEMPORAL!", median_blur, radius)

    radius = to_arr(radius)

    if (len((rs := set(radius))) == 1 and rs.pop() == 1) and mode == ConvMode.SQUARE:
        return clip.std.Median(planes=planes)

    expr_plane = list[list[str]]()

    for r in radius:
        expr_passes = list[str]()

        for mat in ExprOp.matrix('x', r, mode, [(0, 0)]):
            rb = len(mat) + 1
            st = rb - 1
            sp = rb // 2 - 1
            dp = st - 2

            expr_passes.append(f"{mat} sort{st} swap{sp} min! swap{sp} max! drop{dp} x min@ max@ clip")

        expr_plane.append(expr_passes)

    for e in zip(*expr_plane):
        clip = norm_expr(clip, e, planes, func=median_blur)

    return clip


def bilateral(
    clip: vs.VideoNode, ref: vs.VideoNode | None = None, sigmaS: float | list[float] | None = None,
    sigmaR: float | list[float] | None = None, algorithm: int | None = None, pbfic_num: int | list[int] | None = None,
    radius: int | list[int] | None = None, device_id: int | None = None, num_streams: int | None = None,
    use_shared_memory: bool | None = None, block_size: int | tuple[int, int] | None = None,
    backend: BilateralBackend = BilateralBackend.CPU, planes: PlanesT = None,
) -> vs.VideoNode:
    func = FunctionUtil(clip, bilateral, planes)

    if backend == BilateralBackend.CPU:
        bilateral_args = KwargsNotNone(
            ref=ref, sigmaS=sigmaS, sigmaR=sigmaR, planes=func.norm_planes, algorithm=algorithm, PBFICnum=pbfic_num
        )
    else:
        bilateral_args = KwargsNotNone(
            sigma_spatial=sigmaS,
            sigma_color=sigmaR,
            radius=radius,
            device_id=device_id,
            num_streams=num_streams,
            use_shared_memory=use_shared_memory,
            ref=ref,
        )

        if backend == BilateralBackend.GPU_RTC:
            block_x, block_y = normalize_seq(block_size, 2)

            bilateral_args |= KwargsNotNone(block_x=block_x, block_y=block_y)

    return func.return_clip(getattr(func.work_clip, backend).Bilateral(**bilateral_args))


def flux_smooth(
    clip: vs.VideoNode, temporal_threshold: float = 7.0, spatial_threshold: float = 0.0,
    scalep: bool = True, planes: PlanesT = None
) -> vs.VideoNode:
    func = FunctionUtil(clip, flux_smooth, planes)

    if spatial_threshold:
        smoothed = func.work_clip.zsmooth.FluxSmoothST(temporal_threshold, spatial_threshold, scalep)
    else:
        smoothed = func.work_clip.zsmooth.FluxSmoothT(temporal_threshold, scalep)

    return func.return_clip(smoothed)
