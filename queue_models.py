class QueueModel:
    def __init__(self, arrival_rate, service_time, servers=1, Ca2=1.0, Cs2=1.0):
        self.arrival_rate = arrival_rate
        self.service_time = service_time
        self.servers = servers
        self.Ca2 = Ca2
        self.Cs2 = Cs2

    def gg1_delay(self):
        mu = 1.0 / self.service_time
        rho = self.arrival_rate / mu
        if rho >= 1.0:
            return float('inf')
        Wq = (self.service_time * rho / (1 - rho)) * ((self.Ca2 + self.Cs2) / 2.0)
        return Wq

    def gg_c_delay(self):
        mu = 1.0 / self.service_time
        a = self.arrival_rate / mu
        rho = self.arrival_rate / (self.servers * mu)
        if rho >= 1.0:
            return float('inf')

        # Compute Erlang B recursively, then convert to Erlang C. This avoids
        # overflow from directly evaluating a**c / c! for hundreds of bays.
        erlang_b = 1.0
        for n in range(1, self.servers + 1):
            erlang_b = (a * erlang_b) / (n + a * erlang_b)
        P_wait = erlang_b / (1.0 - rho + rho * erlang_b)

        Wq_mm_c = (P_wait * self.service_time) / (self.servers * (1 - rho))
        Wq_gg_c = Wq_mm_c * ((self.Ca2 + self.Cs2) / 2.0)
        return Wq_gg_c

    def waiting_time(self):
        if self.servers == 1:
            return self.gg1_delay()
        else:
            return self.gg_c_delay()

def node_time(self):
    """Total time at node = service_time + wait_time"""
    Wq = self.gg1_delay() if self.servers == 1 else self.gg_c_delay()
    return self.service_time + Wq

def edge_time(self, traverse_time, lanes):
    """Total time on edge = traverse_time + wait_time"""
    temp_model = QueueModel(self.arrival_rate, traverse_time, servers=lanes, 
                             Ca2=self.Ca2, Cs2=self.Cs2)
    Wq = temp_model.gg_c_delay() if lanes > 1 else temp_model.gg1_delay()
    return traverse_time + Wq
