
clear all
clc
close all

%%
data = importdata('telemetry.csv');
mixands = 2; %8
enlarge_pdf = 1.5;

i=1;
NUM = size(data,1);
time = data(:,i); i=i+1;
x = data(:,i); i=i+1;
y = data(:,i); i=i+1;

mixands_state = zeros(NUM*mixands, 2);
mixands_var = zeros(NUM*mixands, 4);
mixands_weight = zeros(NUM*mixands, 1);
for j = 1:mixands
    mixands_state(j:mixands:end, :) = ...
        [data(:,i), data(:,i+1)]; i=i+2;
    mixands_var(j:mixands:end, :) = ...
        [data(:,i), data(:,i+1), data(:,i+2), data(:,i+3)]; i=i+4;
    mixands_weight(j:mixands:end, :) = data(:,i); i=i+1;
end

sensor = data(:,i); i=i+1;

%%
figure(1);
N = 100;
corridor_span = linspace(-3,3,N);
k = 1;
mixture_pdf_history = zeros(length(time), length(corridor_span));
for i = 1:length(time)
    mixture_pdf = zeros(size(corridor_span));
    for j = 1:mixands
        mu = mixands_state(k,2);
        sigma = mixands_var(k, 4);
        mixture_pdf = mixture_pdf + normpdf(corridor_span,mu,sigma);
        k = k + 1;
    end
    mixture_pdf = enlarge_pdf * mixture_pdf / sum(mixture_pdf);
    mixture_pdf_history(i,:) = mixture_pdf;
    plot(mixture_pdf+x(i), corridor_span, 'Color', [0., 0., 1.]); 
    hold on; grid on;
    xlabel('x [m]'); ylabel('y [m]');
    plot(x(i), y(i), 's', 'MarkerSize', 12, 'MarkerFaceColor', [0,0,0], 'MarkerEdgeColor', [0,0,0]);
    plot(mixands_state(i*mixands,1), mixands_state(i*mixands,2), 's', 'MarkerSize', 12, 'MarkerFaceColor', [1,0,0], 'MarkerEdgeColor', [1,0,0]);
    plot(x(i), sensor(i), 'h', 'MarkerSize', 12, 'MarkerFaceColor', [0,0.7,0], 'MarkerEdgeColor', [0,0.7,0]);
    axis([0, 5, -3, 3]); 
    p = 8;
    f = 0.8 - p/(p+2);
    for m = 1:p
        ff= f+m/(p+6);
        if(i-m < 1)
            break
        end
        plot(mixture_pdf_history(i-m,:)+x(i-m), corridor_span, ...
            'Color', [ff, ff, 1.]);
        plot(x(i-m), y(i-m), 's', 'MarkerSize', 12, ...
            'MarkerFaceColor', [ff,ff,ff], 'MarkerEdgeColor', [ff,ff,ff]);
        plot(mixands_state((i-m)*mixands,1), mixands_state((i-m)*mixands,2), 's', 'MarkerSize', 12, ...
            'MarkerFaceColor', [1,ff,ff], 'MarkerEdgeColor', [1,ff,ff]);
%         plot(x(i-m), sensor(i-m), 'h', 'MarkerSize', 12, 'MarkerFaceColor', [ff,0.7,ff], 'MarkerEdgeColor', [0,0.7,0]);
    end
    title(['t=' num2str(time(i))]);
    pause(0.1);
    hold off
end



