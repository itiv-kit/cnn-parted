import os
import numpy as np
from pymoo.core.result import Result
import matplotlib.pyplot as plt

class Optimizer():
    def optimize(self):
        pass


    def _get_paretos_int(self, res : Result) -> list:
        if res.F is None:
            print("### [Optimizer] No solutions found for the given constraints! ###")
            quit()

        X = res.X
        F = res.F
        G = res.G

        data = []
        for i in range(0, len(X)):
            data.append(np.append(G[i], np.append(X[i], F[i])))

        if res.history is not None:
            for h in res.history:
                for ind in h.pop:
                    if np.max(ind.get("G")) > 0:
                        continue
                    data.append(np.append(ind.get("G").tolist(), np.append(ind.get("X").tolist(), ind.get("F").tolist())))
        data = np.unique(data, axis=0)

        g_len = len(G[0])
        x_len = len(X[0])
        comp_hist = np.delete(data, np.s_[0:x_len+g_len], axis=1)
        paretos = self._is_pareto_efficient(comp_hist)
        paretos = np.expand_dims(paretos, 1)
        data = np.hstack([data, paretos])

        return data


    # https://stackoverflow.com/a/40239615
    def _is_pareto_efficient(self, costs : np.ndarray, return_mask : bool = True) -> np.ndarray:
        is_efficient = np.arange(costs.shape[0])
        n_points = costs.shape[0]
        next_point_index = 0  # Next index in the is_efficient array to search for
        while next_point_index<len(costs):
            nondominated_point_mask = np.any(costs<costs[next_point_index], axis=1)
            nondominated_point_mask[next_point_index] = True
            is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
            costs = costs[nondominated_point_mask]
            next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
        if return_mask:
            is_efficient_mask = np.zeros(n_points, dtype = bool)
            is_efficient_mask[is_efficient] = True
            return is_efficient_mask
        else:
            return is_efficient

    def _plot_history(self, res: Result, out_path: str):
        best_results = []
        all_results = []

        if res.history:
            for hist in res.history:
                all_costs = [float(indiv.F) for indiv in hist.pop]
                best_results.append(min(all_costs))
                all_results.append(all_costs)

        fig, ax = plt.subplots(dpi=1200)
        ax.set_xlabel("Generation")
        ax.set_ylabel("EDP")
        for idx, res in enumerate(all_results): 
            ax.scatter(np.full_like(res, idx, dtype=int).tolist(), res, color='b')
        ax.plot(best_results, 'r+')
        os.makedirs(os.path.join(out_path, "dse_results", "figures"), exist_ok=True)
        out_file = os.path.join(out_path, "dse_results", "figures", "history.png")
        fig.savefig(out_file)
