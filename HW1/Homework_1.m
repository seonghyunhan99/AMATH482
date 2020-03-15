%% AMATH 482 Homework 1
% Seong Hyun Han
% 1/24/20

%% Section I
%Starter code with modification

clear; close all; clc;
load Testdata
[data_num, N] = size(Undata); % set data_num to the number of signals obtained 
L=15; % computational spatial domain
n=64; % number of Fourier modes (2^n)
x2=linspace(-L,L,n+1); % define the spatial domain discretization
x=x2(1:n); % only consider the first n points for periodicity  
y=x; z=x; % set spatial domain discretization for y and z axes
k=(2*pi/(2*L))*[0:(n/2-1) -n/2:-1]; % frequency components of FFT rescaled to 2pi domain  
ks=fftshift(k); % Fourier components with zero at the center  

[X,Y,Z]=meshgrid(x,y,z); % Define Cartesian grid in 3D space 
[Kx,Ky,Kz]=meshgrid(ks,ks,ks); % Define 3D grid for the frequency domain 

%% Section II
% (Problem 1) Average the 20 3D spatial data in frequency to reduce noise and find the central frequencies 
ut_ave = zeros(n,n,n);
for j=1:data_num
Un(:,:,:)=reshape(Undata(j,:),n,n,n); % reshape Undata array into a 3D nxnxn array
ut_ave = ut_ave + fftn(Un);
end

ut_ave = ut_ave/data_num; % average of the Fourier transformed signal
ut_ave_fft = fftshift(ut_ave); % fftshifted ut_ave 
ut_ave_abs = abs(ut_ave); % absolute value of ut_ave  

% Find the maximum value in ut_ave_fft and its corresponding index value  
[M,I] = max(abs(ut_ave_fft(:)));
% Find subscript values equivalent to the found index (I) for a nxnxn 3D array   
[r,c,p] = ind2sub(size(ut_ave_fft),I); 

% using subscripts to find the correlated central frequencies
kxo = Kx(r,c,p); % center-frequency for Kx-axis
kyo = Ky(r,c,p); % center-frequency for Ky-axis
kzo = Kz(r,c,p); % center-frequency for Kz-axis

% plot of the absoluate values of ut_ave with values equal to 230  
figure(1)
isosurface(Kx,Ky,Kz,fftshift(ut_ave_abs),230)
axis([1 2 -2 0 -1 0.5]), grid on, drawnow
title('Averaged Data Isosurface Plot - 3D Frequency Domain')
xlabel('Kx')
ylabel('Ky')
zlabel('Kz')
saveas(figure(1),'AMATH482_fig1.png');
print(gcf,'AMATH482_fig1.png','-dpng','-r600');

%% Section III
tau = 0.2; % filter bandwidth 
% 3D Gaussian filter with all filter bandwidths equal to tau 
filter=exp(-tau*(((Kx-kxo).^2)+((Ky-kyo).^2)+((Kz-kzo).^2)));

% Define variables to store 20 x,y,z locations 
x_loc = zeros(1,data_num);
y_loc = zeros(1,data_num);
z_loc = zeros(1,data_num);

% Determine the marble path 
for j=1:data_num
Un(:,:,:)=reshape(Undata(j,:),n,n,n);
ut_filter = filter.*fftshift(fftn(Un)); % filter signal in frequency 
u_filter = abs(ifftn(ut_filter)); % inverse fft of filtered signal into spatial domain

% Determine the marble position based on largest signal in spatial domain
[Max,Idx] = max(u_filter(:));
[row,col,pag] = ind2sub(size(abs(ut_ave)),Idx);

% using subscripts to find the correlated marble locations
x_loc(1,j) = X(row,col,pag);
y_loc(1,j) = Y(row,col,pag);
z_loc(1,j) = Z(row,col,pag);

end

labels = {'t=1','t=2','t=3','t=4','t=5','t=6','t=7','t=8','t=9','t=10','t=11','t=12','t=13','t=14','t=15','t=16','t=17','t=18','t=19','t=20'};

% plot marble path 
figure(2)
plot3(x_loc,y_loc,z_loc), grid on
hold on
plot3(x_loc(1,1), y_loc(1,1), z_loc(1,1),'go')
plot3(x_loc(1,20), y_loc(1,20), z_loc(1,20),'ro')
title('Marble Path - 3D Spatial Domain')
xlabel('X')
ylabel('Y')
zlabel('Z')
text(x_loc,y_loc,z_loc,labels,'FontSize',7)
saveas(figure(2),'AMATH482_fig2.png');
print(gcf,'AMATH482_fig2.png','-dpng','-r600');

%% Section IV
% marble position at the 20th data measurement 
breakup_loc = [x_loc(1,20), y_loc(1,20), z_loc(1,20)];
