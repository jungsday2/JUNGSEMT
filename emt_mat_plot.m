load('emt_results.mat');
t_ms = time * 1000;

% 실제값 변환을 위한 Base Voltage (예: Kundur 24kV L-L RMS -> L-N Peak)
V_BASE_KV = 24.0 * sqrt(2/3); % 약 19.596 kV
PLOT_IN_KV = true; % true면 실제 kV 단위로, false면 pu로 출력

v_mult = 1.0;
if PLOT_IN_KV
    v_mult = V_BASE_KV;
end

figure;
subplot(2,3,1); plot(t_ms, rad2deg(delta));   title('\delta [deg]'); grid on;
subplot(2,3,2); plot(t_ms, omega*100);         title('\Delta\omega [%]'); grid on;
subplot(2,3,3); plot(t_ms, Te);                title('T_e'); grid on;
subplot(2,3,4); plot(t_ms, psi_dd, t_ms, psi_qq); title("\psi''"); legend("d","q"); grid on;
subplot(2,3,5); plot(t_ms, V_bus2' * v_mult);           
if PLOT_IN_KV
    title('V_{bus2} abc [kV (L-N Peak)]'); 
else
    title('V_{bus2} abc [pu]'); 
end
grid on;

subplot(2,3,6); plot(t_ms, V_bus3' * v_mult);           
if PLOT_IN_KV
    title('V_{bus3} abc [kV (L-N Peak)]'); 
else
    title('V_{bus3} abc [pu]'); 
end
grid on;

% 이벤트 시각 표시 (각 axes에 적용)
% xline(t_event*1000, '--r'); 