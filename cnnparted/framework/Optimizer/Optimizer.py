import numpy as np
from pymoo.core.result import Result

class Optimizer():
    def optimize(self):
        pass


    def _get_paretos_int(self, res : Result) -> list:
        if res.F is None:
            print("No solutions found for the given constraints.")
            return

        X = np.round(res.X)
        F = res.F

        data = []
        for i in range(0, len(X)):
            data.append(np.append(X[i], F[i]))

        for h in res.history or []:
            for ind in h.pop:
                if ind.get("G") > 0:
                    continue
                data.append(np.append(np.round(ind.get("X").tolist()), ind.get("F").tolist()))
        data = np.unique(data, axis=0)

        x_len = len(X[0])
        comp_hist = np.delete(data, np.s_[0:x_len], axis=1)
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