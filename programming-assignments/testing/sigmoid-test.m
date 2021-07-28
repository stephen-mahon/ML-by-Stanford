z = linspace(-5, 5, 400);
sigmoid = 1./(1+exp(-z));

figure(1), clf
plot(z, sigmoid, 'linew', 3)
hold on
xlabel('z'),ylabel('g(z)')