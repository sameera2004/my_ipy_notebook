"""

This module is for functions specific to time correlation

"""
from __future__ import absolute_import, division, print_function
from skbeam.core.utils import multi_tau_lags
from skbeam.core.roi import extract_label_indices
from collections import namedtuple
import numpy as np

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

import logging
logger = logging.getLogger(__name__)


def _validate_and_transform_inputs(num_bufs, num_levels, labels):
    """
    This is a helper function to validate inputs and create initial state
    inputs for both one time and two time correlation

    Parameters
    ----------
    num_bufs : int
    num_levels : int
    labels : array
        labeled array of the same shape as the image stack;
        each ROI is represented by a distinct label (i.e., integer)

    Returns
    -------
    label_array : array
        labels of the required region of interests(ROI's)
    pixel_list : array
        1D array of indices into the raveled image for all
        foreground pixels (labeled nonzero)
        e.g., [5, 6, 7, 8, 14, 15, 21, 22]
    num_rois : int
        number of region of interests (ROI)
    num_pixels : array
        number of pixels in each ROI
    lag_steps : array
        the times at which the correlation was computed
    buf : array
        image data for correlation
    img_per_level : array
        to track how many images processed in each level
    track_level : array
        to track processing each level
    cur : array
        to increment the buffer
    norm : dict
        to track bad images
    lev_len : array
        length of each levels
    """
    if num_bufs % 2 != 0:
        raise ValueError("There must be an even number of `num_bufs`. You "
                         "provided %s" % num_bufs)
    label_array = []
    pixel_list = []

    label_array, pixel_list = extract_label_indices(labels)
    # map the indices onto a sequential list of integers starting at 1
    label_mapping = {label: n+1
                     for n, label in enumerate(np.unique(label_array))}
    # remap the label array to go from 1 -> max(_labels)
    for label, n in label_mapping.items():
        label_array[label_array == label] = n

    # number of ROI's
    num_rois = len(label_mapping)

    # stash the number of pixels in the mask
    num_pixels = np.bincount(label_array)[1:]

    # Convert from num_levels, num_bufs to lag frames.
    tot_channels, lag_steps, dict_lag = multi_tau_lags(num_levels, num_bufs)

    # these norm and lev_len will help to find the one time correlation
    # normalization norm will updated when there is a bad image
    norm = {key: [0] * len(dict_lag[key]) for key in (dict_lag.keys())}
    lev_len = np.array([len(dict_lag[i]) for i in (dict_lag.keys())])

    # Ring buffer, a buffer with periodic boundary conditions.
    # Images must be keep for up to maximum delay in buf.
    # separate buffer for each ROI
    buf = {}
    for i in range(num_rois):
        buf[i] = np.zeros((num_levels, num_bufs,
                           len(pixel_list[label_array == i+1])),
                          dtype=np.float64)
    #buf = np.zeros((num_levels, num_bufs, len(pixel_list)),
    #              dtype=np.float64)
    # to track how many images processed in each level
    img_per_level = np.zeros(num_levels, dtype=np.int64)
    # to track which levels have already been processed
    track_level = np.zeros(num_levels, dtype=bool)
    # to increment buffer
    cur = np.ones(num_levels, dtype=np.int64)

    return (label_array, pixel_list, num_rois, num_pixels,
            lag_steps, buf, img_per_level, track_level, cur,
            norm, lev_len)


results = namedtuple(
    'correlation_results',
    ['g2', 'lag_steps', 'internal_state']
)

_internal_state = namedtuple(
    'correlation_state',
    ['buf',
     'G',
     'past_intensity',
     'future_intensity',
     'img_per_level',
     'label_array',
     'track_level',
     'cur',
     'pixel_list',
     'num_pixels',
     'lag_steps',
     'norm',
     'lev_len']
)


def _init_state_one_time(num_levels, num_bufs, labels):
    """Initialize a stateful namedtuple for the generator-based multi-tau
     for one time correlation

    Parameters
    ----------
    num_levels : int
    num_bufs : int
    labels : array
        Two dimensional labeled array that contains ROI information

    Returns
    -------
    internal_state : namedtuple
        The namedtuple that contains all the state information that
        `lazy_one_time` requires so that it can be used to pick up
         processing after it was interrupted
    """
    (label_array, pixel_list, num_rois, num_pixels, lag_steps, buf,
     img_per_level, track_level, cur, norm,
     lev_len) = _validate_and_transform_inputs(num_bufs, num_levels, labels)

    # G holds the un normalized auto- correlation result. We
    # accumulate computations into G as the algorithm proceeds.
    G = np.zeros((num_rois, (num_levels + 1) * num_bufs / 2),
                 dtype=np.float64)
    # matrix for normalizing G into g2
    past_intensity = np.zeros_like(G)
    # matrix for normalizing G into g2
    future_intensity = np.zeros_like(G)

    return _internal_state(
        buf,
        G,
        past_intensity,
        future_intensity,
        img_per_level,
        label_array,
        track_level,
        cur,
        pixel_list,
        num_pixels,
        lag_steps,
        norm,
        lev_len,
    )


def one_time(image_iterable, num_levels, num_bufs, labels,
             internal_state=None):
    if internal_state is None:
        internal_state = _init_state_one_time(num_levels, num_bufs, labels)
    # create a shorthand reference to the results and state named tuple
    s = internal_state

    # divide up vectors
    comm.Scatter(s.label_array, label_array, root=0)

    print (label_array)


def lazy_one_time(image_iterable, num_levels, num_bufs, labels,
                  internal_state=None):
    """Generator implementation of 1-time multi-tau correlation

    If you do not want multi-tau correlation, set num_levels to 1 and
    num_bufs to the number of images you wish to correlate

    Parameters
    ----------
    image_iterable : iterable of 2D arrays
    num_levels : int
        how many generations of downsampling to perform, i.e., the depth of
        the binomial tree of averaged frames
    num_bufs : int, must be even
        maximum lag step to compute in each generation of downsampling
    labels : array
        Labeled array of the same shape as the image stack.
        Each ROI is represented by sequential integers starting at one.  For
        example, if you have four ROIs, they must be labeled 1, 2, 3,
        4. Background is labeled as 0
    internal_state : namedtuple, optional
        internal_state is a bucket for all of the internal state of the
        generator. It is part of the `results` object that is yielded from
        this generator

    Yields
    ------
    namedtuple
        A `results` object is yielded after every image has been processed.
        This `reults` object contains, in this order:
        - `g2`: the normalized correlation
          shape is (len(lag_steps), num_rois)
        - `lag_steps`: the times at which the correlation was computed
        - `_internal_state`: all of the internal state. Can be passed back in
          to `lazy_one_time` as the `internal_state` parameter

    Notes
    -----
    The normalized intensity-intensity time-autocorrelation function
    is defined as

    .. math::
        g_2(q, t') = \\frac{<I(q, t)I(q, t + t')> }{<I(q, t)>^2}

        t' > 0

    Here, ``I(q, t)`` refers to the scattering strength at the momentum
    transfer vector ``q`` in reciprocal space at time ``t``, and the brackets
    ``<...>`` refer to averages over time ``t``. The quantity ``t'`` denotes
    the delay time

    This implementation is based on published work. [1]_

    References
    ----------
    .. [1] D. Lumma, L. B. Lurio, S. G. J. Mochrie and M. Sutton,
        "Area detector based photon correlation in the regime of
        short data batches: Data reduction for dynamic x-ray
        scattering," Rev. Sci. Instrum., vol 70, p 3274-3289, 2000.
    """

    if internal_state is None:
        internal_state = _init_state_one_time(num_levels, num_bufs, labels)
    # create a shorthand reference to the results and state named tuple
    s = internal_state

    # iterate over the images to compute multi-tau correlation
    for image in image_iterable:
        # Compute the correlations for all higher levels.
        level = 0

        # increment buffer
        s.cur[0] = (1 + s.cur[0]) % num_bufs

        # Put the ROI pixels into the ring buffer.
        # have to separate for each roi
        for i in range(s.num_rois):
            s.buf[0, s.cur[0] - 1][i] = np.ravel(image)[s.pixel_list[s.label_array == i+1]]
        #s.buf[0, s.cur[0] - 1] = np.ravel(image)[s.pixel_list]
        buf_no = s.cur[0] - 1
        # Compute the correlations between the first level
        # (undownsampled) frames. This modifies G,
        # past_intensity, future_intensity,
        # and img_per_level in place!
        for i in range(num_rois):
            print (i)
            if rank == i:
                print (i)
                _one_time_process(s.buf[i], s.G[i], s.past_intensity[i],
                                  s.future_intensity[i],
                                  s.label_array[label_array == i+1],
                                  num_bufs, s.num_pixels[i], s.img_per_level,
                                  level, buf_no, s.norm, s.lev_len)

                # check whether the number of levels is one, otherwise
                # continue processing the next level
                processing = num_levels > 1

                level = 1
                while processing:
                    if not s.track_level[level]:
                        s.track_level[level] = True
                        processing = False
                    else:
                        prev = (1 + (s.cur[level - 1] - 2) % num_bufs)
                        s.cur[level] = (
                        1 + s.cur[level] % num_bufs)

                        s.buf[level, s.cur[level] - 1] = ((
                            s.buf[level - 1, prev - 1] +
                            s.buf[level - 1, s.cur[level - 1] - 1]) / 2)

                        # make the track_level zero once that level is processed
                        s.track_level[level] = False

                        # call processing_func for each multi-tau level greater
                        # than one. This is modifying things in place. See comment
                        # on previous call above.
                        buf_no = s.cur[level] - 1
                        _one_time_process(s.buf[i], s.G[i], s.past_intensity[i],
                                  s.future_intensity[i],
                                  s.label_array[label_array == i+1],
                                  num_bufs, s.num_pixels[i], s.img_per_level,
                                  level, buf_no, s.norm, s.lev_len)
                        level += 1

                        # Checking whether there is next level for processing
                        processing = level < num_levels

        # If any past intensities are zero, then g2 cannot be normalized at
        # those levels. This if/else code block is basically preventing
        # divide-by-zero errors.
        if len(np.where(s.past_intensity == 0)[0]) != 0:
            g_max = np.where(s.past_intensity == 0)[0][0]
        else:
            g_max = s.past_intensity.shape[0]

        g2 = (s.G[:g_max] / (s.past_intensity[:g_max] *
                             s.future_intensity[:g_max]))
        yield results(g2, s.lag_steps[:g_max], s)


def _one_time_process(buf, G, past_intensity_norm, future_intensity_norm,
                      label_array, num_bufs, num_pixels, img_per_level,
                      level, buf_no, norm, lev_len):
    """Reference implementation of the inner loop of multi-tau one time
    correlation

    This helper function calculates G, past_intensity_norm and
    future_intensity_norm at each level, symmetric normalization is used.

    .. warning :: This modifies inputs in place.

    Parameters
    ----------
    buf : array
        image data array to use for correlation
    G : array
        matrix of auto-correlation function without normalizations
    past_intensity_norm : array
        matrix of past intensity normalizations
    future_intensity_norm : array
        matrix of future intensity normalizations
    label_array : array
        labeled array where all nonzero values are ROIs
    num_bufs : int, even
        number of buffers(channels)
    num_pixels : array
        number of pixels in certain ROI's
        ROI's, dimensions are : [number of ROI's]X1
    img_per_level : array
        to track how many images processed in each level
    level : int
        the current multi-tau level
    buf_no : int
        the current buffer number
    norm : dict
        to track bad images
    lev_len : array
        length of each level

    Notes
    -----
    .. math::
        G = <I(\tau)I(\tau + delay)>
    .. math::
        past_intensity_norm = <I(\tau)>
    .. math::
        future_intensity_norm = <I(\tau + delay)>
    """
    img_per_level[level] += 1
    # in multi-tau correlation, the subsequent levels have half as many
    # buffers as the first
    i_min = num_bufs // 2 if level else 0
    for i in range(i_min, min(img_per_level[level], num_bufs)):
        # compute the index into the autocorrelation matrix
        t_index = level * num_bufs / 2 + i
        delay_no = (buf_no - i) % num_bufs

        # get the images for correlating
        past_img = buf[level, delay_no]
        future_img = buf[level, buf_no]

        # find the normalization that can work both for bad_images
        #  and good_images
        ind = int(t_index - lev_len[:level].sum())
        normalize = img_per_level[level] - i - norm[level+1][ind]

        # take out the past_ing and future_img created using bad images
        # (bad images are converted to np.nan array)
        if np.isnan(past_img).any() or np.isnan(future_img).any():
            norm[level + 1][ind] += 1
        else:
            for w, arr in zip([past_img*future_img, past_img, future_img],
                              [G, past_intensity_norm, future_intensity_norm]):
                binned = np.bincount(label_array, weights=w)[1:]
                arr[t_index] += ((binned / num_pixels -
                                  arr[t_index]) / normalize)
    return None  # modifies arguments in place!


if __name__ == "__main__":
    import skbeam.core.roi as roi
    inner_radius = 3
    width = 1
    spacing = 1
    num_rings = 2
    center = (13, 14)

    num_levels = 5
    num_bufs = 4  # must be even
    xdim = 25
    ydim = 25

    stack_size = 10
    synthetic_data = np.random.randint(1, 10, (stack_size, xdim, ydim))

    edges = roi.ring_edges(inner_radius, width, spacing, num_rings)
    rings = roi.rings(edges, center, (xdim, ydim))

    #(label_array, pixel_list, num_rois, num_pixels, lag_steps, buf,
    # img_per_level, track_level, cur,
    # norm, lev_len) = _validate_and_transform_inputs(num_bufs,
    #                                                 num_levels, rings)
    for i in range(num_rings):
        if rank==i:
            lazy_one_time(synthetic_data, num_levels, num_bufs, rings,
             internal_state=None)





