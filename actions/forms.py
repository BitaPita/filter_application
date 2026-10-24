
# forms.py (updated)
from django import forms

class ImageUploadForm(forms.Form):
    image = forms.ImageField(label='Select an image',required=False)
    filter_choice = forms.ChoiceField(
        label='Filter Type',
        choices=[
            ('gaussian', 'Gaussian Blur'),
            ('box', 'Box Blur (Mean)'),
            ('median', 'Median Filter'),
            ('sharpen', 'Sharpen'),
            ('edge_sobel', 'Edge Detection - Sobel'),
            ('edge_canny', 'Edge Detection - Canny'),
            ('edge_prewitt', 'Edge Detection - Prewitt'),
            ('edge_laplacian', 'Edge Detection - Laplacian'),
            ('edge_log', 'Edge Detection - Laplacian of Gaussian'),
            ('edge_robinson', 'Edge Detection - Robinson Compass'),
            ('histogram_eq', 'Histogram Equalization'),
            ('fft_magnitude', 'Fourier Magnitude Spectrum'),
            ('fft_phase', 'Fourier Phase Spectrum'),
            ('noise_gaussian', 'Add Gaussian Noise'),
            ('noise_sp', 'Add Salt & Pepper Noise'),
            ('noise_periodic', 'Add Periodic Noise (Sine Wave)'),
            ('noise_speckle', 'Add Speckle Noise (Multiplicative)'),
            ('noise_poisson', 'Add Poisson Noise (Shot Noise)'),
                    
            ('clahe', 'Adaptive Hist Equalization (CLAHE)'),
            ('morph_dilation', 'Morphological Dilation'),
            ('morph_erosion', 'Morphological Erosion'),
            ('threshold', 'Otsu Thresholding (B&W)'),
            ('sepia', 'Sepia Tone (Vintage)'),
            ('emboss', 'Emboss (3D Effect)'),
            ('invert', 'Invert Colors (Negative)'),

            ('custom', 'Custom Convolution'),

            ('noise_gaussian_lowpass', 'Gaussian Filter - Low-Pass (Smooth)'),
            ('noise_gaussian_highpass', 'Gaussian Filter - High-Pass (Sharp)'),
            ('noise_gaussian_bandpass', 'Gaussian Filter - Band-Pass'),
            ('noise_gaussian_notch', 'Gaussian Filter - Notch (Band-Stop)'),
            ('butterworth_noise_lowpass', 'Butterworth Filter - Low-Pass'),
            ('butterworth_noise_highpass', 'Butterworth Filter - High-Pass'),
            ('butterworth_noise_bandpass', 'Butterworth Filter - Band-Pass'),
            ('butterworth_noise_notch', 'Butterworth Filter - Notch'),
            ('ideal_noise_lowpass', 'Ideal Filter - Low-Pass'),
            ('ideal_noise_highpass', 'Ideal Filter - High-Pass'),
            ('ideal_noise_bandpass', 'Ideal Filter - Band-Pass'),
            ('ideal_noise_notch', 'Ideal Filter - Notch'),
        ]
    )
    kernel_size = forms.ChoiceField(
        label='Kernel Size',
        choices=[('3', '3x3'), ('5', '5x5'), ('7', '7x7')],
        initial='3'
    )
    custom_kernel = forms.CharField(
        label='Custom Kernel (comma-separated, row-major)',
        required=False,
        help_text='Example for 3x3: 1,1,1,1,1,1,1,1,1',
        widget=forms.TextInput(attrs={'placeholder': '1,1,1,1,1,1,1,1,1'})
    )
    continue_filtering = forms.BooleanField(
        label='Continue filtering with current result',
        required=False,
        initial=True
    )
    edge_threshold = forms.FloatField(
        label='Edge Threshold (for Canny)',
        required=False,
        initial=0.33,
        min_value=0.0,
        max_value=1.0,
        widget=forms.NumberInput(attrs={'step': '0.05'})
    )

    #noise density
    
    noise_sigma = forms.IntegerField(
        label='Gaussian Noise Sigma',
        required=False,
        initial=25,
        min_value=1,
        max_value=100,
        widget=forms.NumberInput(attrs={'step': '1'})
    )
    
    # برای نویز نمک و فلفل (چگالی نویز بین 0 تا 1)
    noise_amount = forms.FloatField(
        label='Salt & Pepper Amount',
        required=False,
        initial=0.04,
        min_value=0.0,
        max_value=1.0,
        widget=forms.NumberInput(attrs={'step': '0.01'})
    )
    
    # برای نویز پریودیک (دامنه موج)
    noise_amplitude = forms.IntegerField(
        label='Periodic Noise Amplitude',
        required=False,
        initial=30,
        min_value=1,
        max_value=100,
        widget=forms.NumberInput(attrs={'step': '1'})
    )
    
    # برای نویز پریودیک (فرکانس موج)
    noise_frequency = forms.FloatField(
        label='Periodic Noise Frequency',
        required=False,
        initial=0.1,
        min_value=0.001,
        max_value=1.0,
        widget=forms.NumberInput(attrs={'step': '0.01'})
    )

    # پارامتر جدید برای نویزهای گوسی فیلتر شده
    noise_d0 = forms.IntegerField(
        label='Cutoff Frequency (D0) - for Gaussian Filtered Noises',
        required=False,
        initial=30,
        min_value=1,
        max_value=200,
        help_text='Higher value = less filtering, Lower value = more filtering',
        widget=forms.NumberInput(attrs={'step': '1'})
    )
    notch_width = forms.IntegerField(
    label='Notch Width',
    required=False,
    initial=10,
    min_value=1,
    max_value=50,
    help_text='Width of the frequency band to remove',
    widget=forms.NumberInput(attrs={'step': '1'})
)
butterworth_order = forms.IntegerField(
    label='Butterworth Order (n)',
    required=False,
    initial=2,
    min_value=1,
    max_value=10,
    help_text='Higher order = sharper transition',
    widget=forms.NumberInput(attrs={'step': '1'})
)