""
class IBR:
    # ---- 시간 영역 ----
    def stamp_G(self, G_matrix):
        pass

    def stamp_history_current(self, I_total):
        pass

    def update_branch(self, V_nodes):
        pass

    def predict_state(self, dt, t):
        pass

    # ---- 위상자 정상상태 (bumpless start) ----
    def stamp_Y_phasor(self, Y_matrix, omega):
        pass

    def stamp_I_phasor(self, I_vector, omega):
        pass

    def initialize_states(self, V_phasor, omega, dt):
        pass