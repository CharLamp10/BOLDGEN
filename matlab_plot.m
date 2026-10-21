clear
clc
close all

addpath('C:\Users\106742\Desktop\SPM\spm12_orig')

% %% 1) White noise
% noise = normrnd(0,1, [800,3]);
% 
% figure
% plot(noise(:,1) + 6, 'LineWidth', 1)
% hold on
% plot(noise(:,2) + 1, 'LineWidth', 1)
% plot(noise(:,3) + -6, 'LineWidth', 1)
% xticks([])
% yticks([])
% 
% exportgraphics(gcf,'white_noise.png','Resolution',600)
% 
% %% 2) Spectral shaping: multiply the FFT by 1/f
% [N, nSig] = size(noise);
% TR = 1;                      % sampling interval in seconds (assumed)
% fs = 1/TR;                   % sampling rate in Hz
% 
% % Frequency axis in FFT ordering (0, positive freqs, then negative freqs)
% if mod(N,2) == 0
%     f = [0:N/2, -(N/2-1):-1]' * fs/N;
% else
%     f = [0:(N-1)/2, -(N-1)/2:-1]' * fs/N;
% end
% f = abs(f);                  % use |f| so the spectrum stays conjugate-symmetric
% 
% % 1/f multiplier (the DC term would be Inf, so set it to 0 -> zero-mean signal)
% mult = zeros(N,1);
% mult(2:end) = 1 ./ sqrt(f(2:end));
% 
% % FFT -> multiply -> inverse FFT
% noiseFFT = fft(noise, [], 1);
% pinkFFT  = noiseFFT .* mult;                 % implicit expansion across columns
% pink     = real(ifft(pinkFFT, [], 1));       % real() removes numerical residue
% 
% % Normalize to unit standard deviation so it plots on the same scale as before
% pink = pink ./ std(pink, 0, 1);
% 
% figure
% plot(pink(:,1) + 6, 'LineWidth', 1)
% hold on
% plot(pink(:,2) + 1, 'LineWidth', 1)
% plot(pink(:,3) + -6, 'LineWidth', 1)
% xticks([])
% yticks([])
% 
% exportgraphics(gcf,'pink_noise.png','Resolution',600)
% 
% %% 3) Estimate and plot the neural dynamics
% A = [-0.5, -0.2,  0;
%       0.4, -0.5, -0.3;
%       0,    0.2, -0.5];  % coupling matrix
% 
% pink = pink';
% states = zeros(nSig, N);
% states(:,1) = pink(:,1);
% for t = 2:N
%     states(:,t) = states(:,t-1) + A * states(:,t-1) + pink(:,t);
% end
% states = states';
% pink = pink';
% 
% states = states ./ std(states, 0, 1);
% 
% figure
% plot(states(:,1) + 6, 'LineWidth', 1)
% hold on
% plot(states(:,2) + 1, 'LineWidth', 1)
% plot(states(:,3) + -6, 'LineWidth', 1)
% xticks([])
% yticks([])
% 
% exportgraphics(gcf,'states.png','Resolution',600)
% 
% %% 4) Plot the 1/f multiplier in the frequency domain
% posIdx = 2:floor(N/2)+1;     % positive frequencies, excluding DC
% 
% figure
% loglog(f(posIdx), mult(posIdx), 'LineWidth', 1)
% xlabel('Frequency (Hz)')
% ylabel('Multiplier (1/f)')
% title('1/f multiplier (log-log axes)')
% 
% exportgraphics(gcf,'one_over_f_multiplier.png','Resolution',600)
% 
% %% 5) Convolve the pink noise with a hemodynamic response function (SPM)
% hrf = spm_hrf(TR);           % canonical HRF sampled at TR (requires SPM on the path)
% 
% bold = zeros(N, nSig);
% for i = 1:nSig
%     tmp = conv(states(:,i), hrf);      % 'full' convolution: length N + length(hrf) - 1
%     bold(:,i) = tmp(1:N);         % trim back to the original length
% end
% 
% % Normalize for plotting on the same scale
% bold = bold ./ std(bold, 0, 1);
% 
% figure
% plot(bold(:,1) + 6, 'LineWidth', 1)
% hold on
% plot(bold(:,2) + 1, 'LineWidth', 1)
% plot(bold(:,3) + -6, 'LineWidth', 1)
% xticks([])
% yticks([])
% 
% exportgraphics(gcf,'bold.png','Resolution',600)






%% 1) White noise
noise = normrnd(0,1, [800,3]);

figure
plot(noise(:,1) + 6, 'LineWidth', 1)
hold on
plot(noise(:,2) + 1, 'LineWidth', 1)
plot(noise(:,3) + -6, 'LineWidth', 1)
xticks([])
yticks([])

exportgraphics(gcf,'white_noise6.png','Resolution',600)

%% 2) Spectral shaping: multiply the FFT by 1/f
[N, nSig] = size(noise);
TR = 1;                      % sampling interval in seconds (assumed)
fs = 1/TR;                   % sampling rate in Hz

% Frequency axis in FFT ordering (0, positive freqs, then negative freqs)
if mod(N,2) == 0
    f = [0:N/2, -(N/2-1):-1]' * fs/N;
else
    f = [0:(N-1)/2, -(N-1)/2:-1]' * fs/N;
end
f = abs(f);                  % use |f| so the spectrum stays conjugate-symmetric

% 1/f multiplier (the DC term would be Inf, so set it to 0 -> zero-mean signal)
mult = zeros(N,1);
mult(2:end) = 1 ./ sqrt(f(2:end));

% FFT -> multiply -> inverse FFT
noiseFFT = fft(noise, [], 1);
pinkFFT  = noiseFFT .* mult;                 % implicit expansion across columns
pink     = real(ifft(pinkFFT, [], 1));       % real() removes numerical residue

% Normalize to unit standard deviation so it plots on the same scale as before
pink = pink ./ std(pink, 0, 1);

figure
plot(pink(:,1) + 6, 'LineWidth', 1)
hold on
plot(pink(:,2) + 1, 'LineWidth', 1)
plot(pink(:,3) + -6, 'LineWidth', 1)
xticks([])
yticks([])


%% 3) Estimate and plot the neural dynamics
A = [-0.5, -0.2,  0;
      0.4, -0.5, -0.3;
      0,    0.2, -0.5];  % coupling matrix

pink = pink';
states = zeros(nSig, N);
states(:,1) = pink(:,1);
for t = 2:N
    states(:,t) = states(:,t-1) + A * states(:,t-1) + pink(:,t);
end
states = states';
pink = pink';

states = states ./ std(states, 0, 1);

figure
plot(states(:,1) + 6, 'LineWidth', 1)
hold on
plot(states(:,2) + 1, 'LineWidth', 1)
plot(states(:,3) + -6, 'LineWidth', 1)
xticks([])
yticks([])

exportgraphics(gcf,'states6.png','Resolution',600)

%% 5) Convolve the pink noise with a hemodynamic response function (SPM)
hrf = spm_hrf(TR);           % canonical HRF sampled at TR (requires SPM on the path)

bold = zeros(N, nSig);
for i = 1:nSig
    tmp = conv(states(:,i), hrf);      % 'full' convolution: length N + length(hrf) - 1
    bold(:,i) = tmp(1:N);         % trim back to the original length
end

% Normalize for plotting on the same scale
bold = bold ./ std(bold, 0, 1);

figure
plot(bold(:,1) + 6, 'LineWidth', 1)
hold on
plot(bold(:,2) + 1, 'LineWidth', 1)
plot(bold(:,3) + -6, 'LineWidth', 1)
xticks([])
yticks([])

exportgraphics(gcf,'bold6.png','Resolution',600)
