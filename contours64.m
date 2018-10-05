clear all, close all, clc;
nx =64-1; ny = 64;
iter_max = 6;%100
for iter = 1:iter_max
k=(iter-1)*1000;
temp  = sprintf('%d.eps', 0.01*k);
% read the output files
file1  = sprintf('/home/vikas/Desktop/mixinglayer_cpp/64out_o3upwind/%d/omg', k);
omg_data = importdata(file1);

% convert to matrix format
omg=zeros(nx,ny); for j=1:ny; for i=1:nx; omg(i,j)=omg_data((j-1)*nx+i); end end

% grid
x =linspace(-1,1,nx);y =linspace(-1,1,ny);[X,Y]= ndgrid(x,y);

% vorticity contours

% figure:
f1 = figure(1);
W = 4; H = 4;
set(f1,'PaperUnits','inches');set(f1,'PaperOrientation','portrait');
set(f1,'PaperSize',[H,W])    ;set(f1,'PaperPosition',[0,0,W,H]);
contourf(X,Y,omg);axis([-1,1,-1,1]);
%caxis([-50,0]);colorbar;
%set(gcf,'colormap', gray);
xlabel('x');ylabel('y');
print(f1,'-deps','-color',temp);
end

