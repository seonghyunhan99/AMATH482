%% AMATH 482 Homework 2
% Seong Hyun Han
% 2/07/20

%% Part 1.1 
% Starter code - Plot a portion of Handel's Messiah  

clear; close all; clc
load handel
% y = change in amplitude (loudness) of the sound wave with respect to time 
v = y'; % Store transposed audio data (y)
% Fs = sample rate; length(v) = total sample size
figure(1)
subplot(2,2,1)
plot((1:length(v))/Fs,v); 
set(gca,'Xlim',[0,length(v)/Fs],'Fontsize',10) 
xlabel('Time (sec)');
ylabel('Amplitude');
title('Signal of Interest, v(n)');
hold on 

% Playback the portion of music 
% p8 = audioplayer(v,Fs);
% playblocking(p8);

% Produce spectrograms of the portion of music using Gabor filtering
% Implementation of the Gabor transform used in class 

duration = length(v)/Fs; % Length of music (8.9249 sec)
L=duration; % Computational time domain
n=length(v); % Number of Fourier modes (2^n)
t2=linspace(0,L,n+1); % Define the time domain discretization
t=t2(1:n); % Only consider the first n points for periodicity 
% Fourier components rescaled to have frequency in hertz 
k=(1/L)*[0:(n-1)/2 (-n+1)/2:-1]; % account for n being odd
ks=fftshift(k); % Fourier components with zero at the center 

% Plot the Gaussian window on the audio signal
tau_ex = L/2;
a = 50;
g = exp(-a*(t-tau_ex).^2);
plot(t,g,'r','Linewidth',1) 

% Frequency plot of the signal of interest
subplot(2,2,3)
plot(ks, fftshift(abs(fft(v))))
xlabel('Frequency (Hz)');
ylabel('Amplitude');
title('FFT of Signal of Interest, v(n)');
set(gca,'Xlim',[-abs(max(ks)), abs(max(ks))],'Fontsize',10) 
% saveas(figure(2),'AMATH482_HW2_fig2.png');
% print(gcf,'AMATH482_HW2_fig2.png','-dpng','-r600');

% Create Gabor transform spectrogram
a1 = 50; % width parameter
tslide=0:0.1:L; % tau = 0.1 
vgt_spec_g = zeros(length(tslide),n); % store filtered frequency data 
for j=1:length(tslide)
    % Gabor filter function / window
    gabor=exp(-a1*(t-tslide(j)).^2); % tau = tslide(j) = translation parameter 
    vg=gabor.*v; % apply filter to signal (mutiplication in time domain) 
    vgt=fft(vg); 
    vgt_spec_g(j,:) = fftshift(abs(vgt)); % We don't want to scale it
end

% Plot spectrogram
subplot(2,2,[2,4])
pcolor(tslide,ks,vgt_spec_g.') 
shading interp 
set(gca,'Fontsize',10) 
colormap(hot)
colorbar
xlabel('Time (sec)');
ylabel('Frequency (Hz)');
title('Spectrogram of Signal of Interest, v(n)');

%% Part 1.2
% Effect of window width of the Gabor transform on the spectrogram

figure(2)
% Spectrograms for varying a (window width) of the Gabor transform
a_vec = [100 10 1]; % a = 100, a = 10, a = 1
for jj = 1:length(a_vec)
    a = a_vec(jj);
    tslide=0:0.1:10;
    vgt_spec = zeros(length(tslide),n);
    for j=1:length(tslide)
        g=exp(-a*(t-tslide(j)).^2); 
        vg=g.*v; 
        vgt=fft(vg); 
        vgt_spec(j,:) = fftshift(abs(vgt)); 
    end

    subplot(2,2,jj)
    pcolor(tslide,ks,vgt_spec.'), 
    shading interp 
    title(['Audio Signal Spectrogram, a = ',num2str(a)],'Fontsize',10)
    set(gca,'Fontsize',10) 
    colormap(hot) 
    colorbar
    xlabel('Time (sec)');
    ylabel('Frequency (Hz)');
    
end

% Plot unfiltered Fourier transformed audio data v 
vgt_spec = repmat(fftshift(abs(fft(v))),length(tslide),1); % format to 90 by length(v) matrix
subplot(2,2,4)
pcolor(tslide,ks,vgt_spec.'), 
shading interp 
title('FFT of Audio Signal','Fontsize',10)
xlabel('Time (sec)');
ylabel('Frequency (Hz)');
set(gca,'Fontsize',10) 
colormap(hot) 
colorbar

%% Part 1.3
% Effect of oversampling versus undersampling on the spectrogram

% Check to see if there is window overlap or not
figure(3)
tau_ex_1 = 1.5;
tau_ex_2 = 3;
tau_ex_3 = 5.5;
tau_ex_4 = 5.6;
a = 50;
g = exp(-a*(t-tau_ex_1).^2);
g2 = exp(-a*(t-tau_ex_2).^2);
g3 = exp(-a*(t-tau_ex_3).^2);
g4 = exp(-a*(t-tau_ex_4).^2);
plot(t,v,'Color',[0, 0.4470, 0.7410],'Linewidth',1)
hold on
plot(t,g,'r',t,g2,'r','Linewidth',1) 
plot(t,g3,'r',t,g4,'r','Linewidth',1)
set(gca,'Xlim',[0 L],'Fontsize',10), xlabel('Time (t)'), ylabel('Amplitude')
title('Audio Signal with Gabor Windows, tau = 1.5 & tau = 0.1');

% Undersampling (windows don't overlap)
a1 = 50; % width parameter
tslide=0:1.5:L; % tau = 1.5
vgt_spec = zeros(length(tslide),n);
for j=1:length(tslide)
    % Gabor filter function / window
    gabor=exp(-a1*(t-tslide(j)).^2); % tau = tslide(j) = translation parameter 
    vg=gabor.*v; % apply filter to signal (mutiplication in time domain) 
    vgt=fft(vg); 
    vgt_spec(j,:) = fftshift(abs(vgt)); % We don't want to scale it
end

% Plot spectrogram
figure(4)
subplot(2,1,1)
pcolor(tslide,ks,vgt_spec.'), 
shading interp 
set(gca,'Fontsize',10) 
colormap(hot)
colorbar
xlabel('Time (sec)');
ylabel('Frequency (Hz)');
title('Undersampling Spectrogram, tau = 1.5');

% oversampling (windows overlap)
a1 = 50; % width parameter
tslide=0:0.1:L; % tau = 0.1
vgt_spec = zeros(length(tslide),n);
for j=1:length(tslide)
    % Gabor filter function / window
    gabor=exp(-a1*(t-tslide(j)).^2); % tau = tslide(j) = translation parameter 
    vg=gabor.*v; % apply filter to signal (mutiplication in time domain) 
    vgt=fft(vg); 
    vgt_spec(j,:) = fftshift(abs(vgt)); % We don't want to scale it
end

% Plot spectrogram
subplot(2,1,2)
pcolor(tslide,ks,vgt_spec.'), 
shading interp 
set(gca,'Fontsize',10) 
colormap(hot)
colorbar
xlabel('Time (sec)');
ylabel('Frequency (Hz)');
title('Oversampling Spectrogram, tau = 0.1');

%% Part 1.4
% Comparison of different Gabor windows: Gaussian window, Mexican hat
% wavelet, and step-function (Shannon) window 

% Mexican hat 
% Plot of Mexican hat wavelet in time 
tau = 5; % filter center point
sigma = 0.1; % width parameter
c1 = (2/(sqrt(3*sigma)*pi.^(1/4))); % window (height) scaling parameter
c = 1; % scaling factor equal to Gaussian window and Shannon window for comparison
% Mexican hat equation 
mexican_hat_ex = c*(1-((t-tau)/sigma).^2).*exp((-(t-tau).^2)/(2*(sigma.^2)));
figure(5)
plot(t, mexican_hat_ex, 'b', 'Linewidth',1)

tau_slide=0:0.1:L; % tau = 0.1
vmt_spec_m = zeros(length(tau_slide),n);
for j=1:length(tau_slide)
    % Mexican hat wavelet function / window
    mexican_hat = c*(1-((t-tau_slide(j))/sigma).^2).*exp((-(t-tau_slide(j)).^2)/(2*(sigma.^2)));  
    vm=mexican_hat.*v; % apply filter to signal (mutiplication in time domain) 
    vmt=fft(vm); 
    vmt_spec_m(j,:) = fftshift(abs(vmt)); % We don't want to scale it
end

% Plot spectrogram using Mexican hat window 
figure(6)
pcolor(tau_slide,ks,vmt_spec_m.'), 
shading interp 
set(gca,'Fontsize',10) 
colormap(hot)
xlabel('Time (sec)');
ylabel('Frequency (Hz)');
title('Handel Spectrogram - Mexican Hat Window');
colorbar

% Shannon filter
w = 0.4; % width parameter 
tau_ex_3 = 5;
% Step function created using two heaviside functions 
step_ex = heaviside((t-(tau_ex_3)+(w/2))) - heaviside((t-(tau_ex_3)-(w/2)));
figure(7)
plot(t, step)

tau_slide=0:0.1:L; % tau = 0.1
vmt_spec_s = zeros(length(tau_slide),n);
for j=1:length(tau_slide)
    % Shannon filter function / window
    step = heaviside((t-(tau_slide(j))+(w/2))) - heaviside((t-(tau_slide(j))-(w/2))); 
    vm=step.*v; % apply filter to signal (mutiplication in time domain) 
    vmt=fft(vm); 
    vmt_spec_s(j,:) = fftshift(abs(vmt)); % We don't want to scale it
end

% Plot spectrogram using Shannon window
figure(8)
pcolor(tau_slide,ks,vmt_spec_s.'), 
shading interp 
set(gca,'Fontsize',12) 
colormap(hot)
xlabel('Time (sec)');
ylabel('Frequency (Hz)');
title('Handel Spectrogram - Shannon Window');
colorbar

% Plot the three spectrograms (three windows)
figure(9)
subplot(3,1,1)
pcolor(tslide,ks,vgt_spec_g.') 
shading interp 
set(gca,'Ylim',[0, max(abs(ks))],'Fontsize',10) 
colormap(hot)
colorbar
xlabel('Time (sec)');
ylabel('Frequency (Hz)');
title('Handel Spectrogram - Gaussian Window');

subplot(3,1,2)
pcolor(tau_slide,ks,vmt_spec_m.'), 
shading interp 
set(gca,'Ylim',[0, max(abs(ks))],'Fontsize',10) 
colormap(hot)
xlabel('Time (sec)');
ylabel('Frequency (Hz)');
title('Handel Spectrogram - Mexican Hat Window');
colorbar

subplot(3,1,3)
pcolor(tau_slide,ks,vmt_spec_s.'), 
shading interp 
set(gca,'Ylim',[0, max(abs(ks))],'Fontsize',10) 
colormap(hot)
xlabel('Time (sec)');
ylabel('Frequency (Hz)');
title('Handel Spectrogram - Shannon Window');
colorbar

% Plot the three windows 
figure(10)
tau_ex_3 = 5;
g3 = exp(-a*(t-tau_ex_3).^2);
plot(t, mexican_hat_ex,'b','Linewidth',1)
hold on 
plot(t,g3,'r','Linewidth',1)
plot(t, step_ex,'k','Linewidth',1)
hold off
xlabel('Time (sec)');
ylabel('Frequency (Hz)');
title('Gaussian, Mexican Hat, and Shannon Windows');
set(gca,'Xlim',[(5)-2, (5)+2],'Fontsize',10) 
legend('Mexican Hat','Gaussian', 'Shannon')
%% Part 2 
% Piano
figure(11)
[y2,Fs] = audioread('music1.wav');
p = y2'; % Store transposed audio data (y)
tr_piano=length(p)/Fs; % record time in seconds
plot((1:length(p))/Fs,p);
xlabel('Time [sec]'); ylabel('Amplitude');
title('Mary had a little lamb (piano)');
% p8 = audioplayer(y,Fs); playblocking(p8);

L=tr_piano; % computational time domain
n=length(p); % number of Fourier modes (2^n)
t2=linspace(0,L,n+1); % define the time domain discretization
t=t2(1:n); % only consider the first n points for periodicity 
k=((1)/L)*[0:n/2-1 -n/2:-1]; % Fourier components rescaled to have frequency in hertz 
ks=fftshift(k); % Fourier components with zero at the center 

% Frequency plot of the music (piano) 
% figure(11)
% plot(ks, fftshift(abs(fft(p))))
% xlabel('Frequency(Hz)');
% ylabel('Amplitude');
% title('FFT of Music 1 - Piano');

a2 = 100; % width parameter
tslide=0:0.1:L; % tau = 0.1
pgt_spec = zeros(length(tslide),n);
for j=1:length(tslide)
    % Gabor filter function / window
    gabor=exp(-a2*(t-tslide(j)).^2); % tau = tslide(j) = translation parameter 
    pg=gabor.*p; % apply filter to signal (mutiplication in time domain) 
    pgt=fft(pg); 
    pgt_spec(j,:) = fftshift(abs(pgt)); % We don't want to scale it
end

% Plot Music 1 piano spectrogram
figure(12)
subplot(1,2,1)
pcolor(tslide,ks,pgt_spec.'), 
xticks(0:1:tr_piano)
yticks(0:200:1300)
shading interp 
set(gca,'Ylim',[0 1300],'Fontsize',10) 
colormap(hot)
colorbar
xlabel('Time (sec)');
ylabel('Frequency (Hz)');
title('Spectrogram of Music 1 - Piano');

% Recorder 

% figure(13)
[y3,Fs] = audioread('music2.wav');
r = y3'; % Store transposed audio data (y)
tr_rec=length(r)/Fs; % record time in seconds
% plot((1:length(r))/Fs,r);
% xlabel('Time [sec]'); ylabel('Amplitude');
% title('Mary had a little lamb (recorder)');
% p8 = audioplayer(y,Fs); playblocking(p8);

L=tr_rec; % computational time domain
n=length(r); % number of Fourier modes (2^n)
t2=linspace(0,L,n+1); % define the time domain discretization
t=t2(1:n); % only consider the first n points for periodicity 
k=(1/L)*[0:n/2-1 -n/2:-1]; % Fourier components rescaled to have frequency in hertz 
ks=fftshift(k); % Fourier components with zero at the center 

% Frequency plot of the music (recorder) 
% figure(14)
% plot(ks, fftshift(abs(fft(r))))
% xlabel('Frequency(Hz)');
% ylabel('Amplitude');
% title('FFT of Music 2 - Recorder');

a3 = 100; % width parameter
tslide=0:0.1:L; % tau = 0.1
rgt_spec = zeros(length(tslide),n);
for j=1:length(tslide)
    % Gabor filter function / window
    gabor=exp(-a3*(t-tslide(j)).^2); % tau = tslide(j) = translation parameter 
    rg=gabor.*r; % apply filter to signal (mutiplication in time domain) 
    rgt=fft(rg); 
    rgt_spec(j,:) = fftshift(abs(rgt)); % We don't want to scale it
end

% Plot Music 2 spectrogram
subplot(1,2,2)
pcolor(tslide,ks,rgt_spec.'), 
xticks(0:1:tr_piano)
yticks(700:200:3200)
shading interp 
set(gca,'Ylim',[0 3200],'Fontsize',10) 
colormap(hot)
colorbar
xlabel('Time (sec)');
ylabel('Frequency (Hz)');
title('Spectrogram of Music 2 - Recorder');

%% Part 2.2 
% Overtone Remover 
% Piano
clear all; close all; clc
figure(13)
[y2,Fs] = audioread('music1.wav');
p = y2'; % Store transposed audio data (y)
tr_piano=length(p)/Fs; % record time in seconds
plot((1:length(p))/Fs,p);
set(gca, 'YGrid', 'off', 'XGrid', 'on')
xlabel('Time [sec]'); ylabel('Amplitude');
title('Mary had a little lamb (piano)');
% p8 = audioplayer(y,Fs); playblocking(p8);

L=tr_piano; % computational time domain
n=length(p); % number of Fourier modes (2^n)
t2=linspace(0,L,n+1); % define the time domain discretization
t=t2(1:n); % only consider the first n points for periodicity 
k=(1/L)*[0:n/2-1 -n/2:-1]; % Fourier components rescaled to have frequency in hertz 
ks=fftshift(k); % Fourier components with zero at the center 

% Frequency plot of the music (recorder) 
% figure(14)
% plot(ks, fftshift(abs(fft(r))))
% xlabel('Frequency(Hz)');
% ylabel('Amplitude');
% title('FFT of Music 2 - Recorder');

a3 = 100; % width parameter
tslide=0:0.1:L;
pgt_spec = zeros(length(tslide),n);
for j=1:length(tslide)
    % Gabor filter function / window
    gabor=exp(-a3.*(t-tslide(j)).^2); % tau = tslide(j) = translation parameter 
    pg=gabor.*p; % apply filter to signal (mutiplication in time domain) 
    pgt=fft(pg); 
    [M,I] = max(abs(pgt)); % find maximum frequency amplitude value and its index
    center_freq = k(I); % define center frequency from index 
    % width parameter = 1
    overtone = exp(-(0.5).*(k-center_freq).^2); % use Gaussian filter around center frequency 
    pgt_final = pgt.*overtone; % apply Gaussian filter in Fourier domain
    pgt_spec(j,:) = fftshift(abs(pgt_final)); % We don't want to scale it
end

% Plot spectrogram
figure(15)
subplot(2,1,1)
pcolor(tslide,ks,pgt_spec.'), 
xticks(0:0.5:tr_piano)
yticks(230:20:340)
shading interp 
set(gca,'Ylim',[230 340],'Fontsize',10) 
colormap(hot)
colorbar
grid on
set(gca,'layer','top')
ax = gca;
ax.GridColor = 'w';
ax.LineWidth = 1;
ax.YGrid = 'off';
xlabel('Time (sec)');
ylabel('Frequency (Hz)');
title('Spectrogram of Overtones Filtered Music 1 - Piano');
colorbar

% Recorder
%clear all; close all; clc
%figure(13)
[y3,Fs] = audioread('music2.wav');
r = y3'; % Store transposed audio data (y)
tr_rec=length(r)/Fs; % record time in seconds
%plot((1:length(r))/Fs,r);
%xlabel('Time [sec]'); ylabel('Amplitude');
%title('Mary had a little lamb (recorder)');
% p8 = audioplayer(y,Fs); playblocking(p8);

L=tr_rec; % computational time domain
n=length(r); % number of Fourier modes (2^n)
t2=linspace(0,L,n+1); % define the time domain discretization
t=t2(1:n); % only consider the first n points for periodicity 
k=(1/L)*[0:n/2-1 -n/2:-1]; % Fourier components rescaled to have frequency in hertz 
ks=fftshift(k); % Fourier components with zero at the center 

% Frequency plot of the music (recorder) 
% figure(14)
% plot(ks, fftshift(abs(fft(r))))
% xlabel('Frequency(Hz)');
% ylabel('Amplitude');
% title('FFT of Music 2 - Recorder');

a3 = 100; % width parameter
tslide=0:0.1:L;
rgt_spec = zeros(length(tslide),n);
for j=1:length(tslide)
    % Gabor filter function / window
    gabor=exp(-a3.*(t-tslide(j)).^2); % tau = tslide(j) = translation parameter 
    rg=gabor.*r; % apply filter to signal (mutiplication in time domain) 
    rgt=fft(rg); 
    [M,I] = max(abs(rgt));
    center_freq = k(I);
    % width parameter = 1
    overtone = exp(-(0.01).*(k-center_freq).^2);
    rgt_final = rgt.*overtone;
    rgt_spec(j,:) = fftshift(abs(rgt_final)); % We don't want to scale it
end

% Plot spectrogram
subplot(2,1,2)
pcolor(tslide,ks,rgt_spec.'), 
xticks(0:0.5:tr_rec)
yticks(700:50:1100)
shading interp 
set(gca,'Ylim',[700 1100],'Fontsize',10) 
colormap(hot)
colorbar
grid on
set(gca,'layer','top')
ax = gca;
ax.GridColor = 'w';
ax.LineWidth = 1;
ax.YGrid = 'off';
xlabel('Time (sec)');
ylabel('Frequency (Hz)');
title('Spectrogram of Overtones Filtered Music 2 - Recorder');


