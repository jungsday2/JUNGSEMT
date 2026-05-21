load('emt_results.mat');
t_ms = time * 1000;

figure;
subplot(2,3,1); plot(t_ms, rad2deg(delta));   title('\delta [deg]'); grid on;
subplot(2,3,2); plot(t_ms, omega*100);         title('\Delta\omega [%]'); grid on;
subplot(2,3,3); plot(t_ms, Te);                title('T_e'); grid on;
subplot(2,3,4); plot(t_ms, psi_dd, t_ms, psi_qq); title("\psi''"); legend("d","q"); grid on;
subplot(2,3,5); plot(t_ms, V_bus2');           title('V_{bus2} abc'); grid on;
subplot(2,3,6); plot(t_ms, V_bus3');           title('V_{bus3} abc'); grid on;

xline(t_event*1000, '--r');   % 이벤트 시각 표시 (각 axes에 적용)