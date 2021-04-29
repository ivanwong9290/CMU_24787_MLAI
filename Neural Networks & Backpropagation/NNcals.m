W1 = [1 2 -3 0 1 -3; 3 1 2 1 0 2; 2 2 2 2 2 1; 1 0 2 1 -2 2];
W2 = [1 2 -2 1; 1 -1 1 2; 3 1 -1 1];
x0 = [1 1 0 0 1 1]';
t = [0 1 0]';
b1 = [1 1 1 1]';
b2 = [1 1 1]';

a1 = W1*x0 + b1

x1 = [1/(1+exp(-2)) 1/(1+exp(-7)) 1/(1+exp(-8)) 1/(1+exp(-2))]'

a2 = W2*x1 + b2


num = exp(a2)
den = sum(num, "all")

x2 = num/den
x2_i = 1./x2

softmax_derivative = [x2(1)*(1-x2(1)) -x2(1)*x2(2) -x2(1)*x2(3);
-x2(2)*x2(1) x2(2)*(1-x2(2)) -x2(2)*x2(3);
-x2(3)*x2(1) -x2(3)*x2(2) x2(3)*(1-x2(3))]

del2 = -softmax_derivative(:, 2)*3.8236

sigmoid_derivative = [x1(1)*(1-x1(1)) x1(2)*(1-x1(2)) x1(3)*(1-x1(3)) x1(4)*(1-x1(4))]'
temp = W2'*del2
del1 = [temp(1)*sigmoid_derivative(1) temp(2)*sigmoid_derivative(2) temp(3)*sigmoid_derivative(3) temp(4)*sigmoid_derivative(4)]'

L = -t(2)*log(x2(2))

W2_new = W2 - 0.5*del2*x1'
b2_new = b2 - 0.5*del2
W1_new = W1 - 0.5*del1*x0'
b1_new = b1 - 0.5*del1

W1 = W1_new;
W2 = W2_new;
b2 = b2_new;
b1 = b1_new;

a1 = W1*x0 + b1

x1 = [1/(1+exp(-2)) 1/(1+exp(-7)) 1/(1+exp(-8)) 1/(1+exp(-2))]'

a2 = W2*x1 + b2


num = exp(a2)
den = sum(num, "all")

x2 = num/den
x2_i = 1./x2

softmax_derivative = [x2(1)*(1-x2(1)) -x2(1)*x2(2) -x2(1)*x2(3);
-x2(2)*x2(1) x2(2)*(1-x2(2)) -x2(2)*x2(3);
-x2(3)*x2(1) -x2(3)*x2(2) x2(3)*(1-x2(3))]

del2 = -softmax_derivative(:, 2)*a2(2)

sigmoid_derivative = [x1(1)*(1-x1(1)) x1(2)*(1-x1(2)) x1(3)*(1-x1(3)) x1(4)*(1-x1(4))]'
temp = W2'*del2
del1 = [temp(1)*sigmoid_derivative(1) temp(2)*sigmoid_derivative(2) temp(3)*sigmoid_derivative(3) temp(4)*sigmoid_derivative(4)]'