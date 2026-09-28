template <typename T>
T im2col_get_pixel(const T* im, int height, int width, int channels,
                        int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 || row >= height || col >= width) return 0;
    return im[col + width * (row + height * channel)];
}

// From Berkeley Vision's Caffe!
// https://github.com/BVLC/caffe/blob/master/LICENSE
template <typename T>
void im2col(const T* data_im, int channels, int height, int width,
            int ksize, int stride, int pad, T* data_col)
{
    int c, h, w;
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;

    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                    im_row, im_col, c_im, pad);
            }
        }
    }
}

// im2col written transposed (out[j * channels * ksize^2 + c] = col[c][j]), for output pixels
// [j_begin, j_end): the CPU GEMM's B operand without the separate transpose of the column matrix
template <typename T>
void im2col_transposed(const T* data_im, int channels, int height, int width, int ksize, int stride,
                       int pad, T* out, int j_begin, int j_end)
{
    const int width_col = (width + 2 * pad - ksize) / stride + 1;
    const int f = channels * ksize * ksize;
    for (int j = j_begin; j < j_end; ++j) {
        const int h = j / width_col, w = j % width_col;
        T* row = out + (size_t)j * f;
        int c = 0;
        for (int c_im = 0; c_im < channels; ++c_im)
            for (int h_offset = 0; h_offset < ksize; ++h_offset)
                for (int w_offset = 0; w_offset < ksize; ++w_offset, ++c)
                    row[c] = im2col_get_pixel(data_im, height, width, channels, h_offset + h * stride,
                                              w_offset + w * stride, c_im, pad);
    }
}
