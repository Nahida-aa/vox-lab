use realfft::num_complex::Complex;
use tch::{Device, Kind, Tensor};

/// Convert a spectrogram from STFT format (flat `[frame × bin]` layout with complex values)
/// to "Complex as Channels" (CaC) format — a tensor of shape `[2, n_fft/2, num_frames]`
/// with separate real and imaginary channels.
///
/// * `spectrogram` is a flat slice of complex values in `[frame × bin]` order, where each frame
///   has `n_fft / 2` bins (Nyquist already dropped by the forward STFT).
/// * `n_fft` is the FFT size used to compute the STFT, which determines the number of bins.
/// * `device` is the target device for the output tensor.
pub fn stft_to_cac(spectrogram: &[Complex<f32>], n_fft: usize, device: Device) -> Tensor {
    let freq_bins = n_fft / 2; // 2048
    let num_frames = spectrogram.len() / freq_bins;

    // Allocate [2, freq_bins, num_frames] row-major
    let mut data = vec![0.0f32; 2 * freq_bins * num_frames];

    for bin in 0..freq_bins {
        for frame in 0..num_frames {
            let c = spectrogram[frame * freq_bins + bin];
            data[bin * num_frames + frame] = c.re;
            data[freq_bins * num_frames + bin * num_frames + frame] = c.im;
        }
    }

    Tensor::from_slice(&data)
        .reshape([2, freq_bins as i64, num_frames as i64])
        .to_kind(Kind::Float)
        .to(device)
}

/// Convert CaC data from a flat `[2, freq_bins, num_frames]` row-major f32 slice
/// to STFT format (flat `[frame × bin]` layout with complex values).
///
/// This is a pure CPU operation.
///
/// * `data` is a flat `&[f32]` of length `2 * freq_bins * num_frames`, laid out as
///   `[reals..., imags...]` where each half is `[freq_bins, num_frames]` row-major.
/// * `freq_bins` is `n_fft / 2`.
/// * `num_frames` is the number of time frames.
///
/// Returns a flat vector of `num_frames × (freq_bins + 1)` complex values, with zeroed
/// Nyquist bins appended to each frame.
pub fn cac_data_to_complex(data: &[f32], freq_bins: usize, num_frames: usize) -> Vec<Complex<f32>> {
    let bins = freq_bins + 1; // Add Nyquist bin
    let mut spectrogram = vec![Complex::new(0.0, 0.0); num_frames * bins];

    let (reals, imags) = data.split_at(freq_bins * num_frames);

    reals
        .iter()
        .zip(imags)
        .map(|(&re, &im)| Complex::new(re, im))
        .enumerate()
        .for_each(|(i, c)| {
            let frame = i % num_frames;
            let bin = i / num_frames;
            spectrogram[frame * bins + bin] = c;
        });

    spectrogram
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_shape() {
        let n_fft = 8;
        let freq_bins = n_fft / 2; // 4
        let num_frames = 3;
        let spec = vec![Complex::new(0.0, 0.0); num_frames * freq_bins];

        let tensor = stft_to_cac(&spec, n_fft, Device::Cpu);

        assert_eq!(tensor.size(), vec![2, 4, 3]);
    }

    #[test]
    fn real_imag_split() {
        let n_fft = 4;
        let spec = vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)];

        let tensor = stft_to_cac(&spec, n_fft, Device::Cpu);
        let flat = tensor.reshape([-1]);
        let data: Vec<f32> = (&flat).try_into().unwrap();

        assert_eq!(data, vec![1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn round_trip_preserves_values() {
        let n_fft = 8;
        let freq_bins = n_fft / 2;
        let num_frames = 4;

        let mut spec: Vec<Complex<f32>> = Vec::with_capacity(num_frames * freq_bins);
        for frame in 0..num_frames {
            for bin in 0..freq_bins {
                spec.push(Complex::new(
                    (frame * 100 + bin) as f32,
                    (frame * 100 + bin + 50) as f32,
                ));
            }
        }

        let tensor = stft_to_cac(&spec, n_fft, Device::Cpu);
        let flat = tensor.reshape([-1]);
        let data: Vec<f32> = (&flat).try_into().unwrap();
        let roundtrip = cac_data_to_complex(&data, freq_bins, num_frames);

        let bins_out = freq_bins + 1;
        assert_eq!(roundtrip.len(), num_frames * bins_out);

        for frame in 0..num_frames {
            for bin in 0..freq_bins {
                let orig = spec[frame * freq_bins + bin];
                let rt = roundtrip[frame * bins_out + bin];
                assert!((orig.re - rt.re).abs() < 1e-6);
                assert!((orig.im - rt.im).abs() < 1e-6);
            }
            let nyquist = roundtrip[frame * bins_out + freq_bins];
            assert_eq!(nyquist.re, 0.0);
            assert_eq!(nyquist.im, 0.0);
        }
    }
}
