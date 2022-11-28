img = imread('lenna.jpg');

img = rgb2gray(img);

y =psf(img); %Apply the psf
Y = fft2(y); %Apply 2D FFT to the altered image
X = fft2(img); %Apply 2D FFT to the original image

B = Y./X; %Find the impulse response of the psf

%Plot the impulse response
figure(1)
imagesc(abs(log(fftshift(B))))
title("Impulse response of the PSF")
colormap gray

%Find the filter using a threshold that will remove the blurring
threshold = 0.9;
for i = 1:257
    for j = 1:255
        if 1/abs(B(i,j)) < threshold
            H(i,j) = 1/B(i,j);
        else
            H(i,j) = threshold * (abs(B(i,j))/B(i,j));
        end
    end
end

%Apply inverse filtering
Fhat = H .* Y;
inv = ifft2(Fhat);

%MSE
mse = (img-uint8(abs(inv))).^2;
mse = mse ./2;

figure(2)
subplot(2,2,1)
title("Original iamge")
imagesc(img);
subplot(2,2,2)
imagesc(y);
title("Image after the PSF")
subplot(2,2,3)
imagesc(abs(inv));
title("Image after reverse filtering")
subplot(2,2,4)
imagesc(mse);
title("MSE")
colormap gray

