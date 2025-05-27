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
        all_constraints = []

        n_obj = res.F.shape[-1]
        n_constr = res.G.shape[-1]
        if res.history:
            for hist in res.history:
                cost_at_time = []
                constr_at_time = []
                for ind in hist.pop:
                    if np.max(ind.get("G")) > 0:
                        cost_at_time.append( np.full((n_obj), np.nan) )
                        constr_at_time.append( np.full((n_constr), np.nan) )
                    else:
                        cost_at_time.append(ind.get("F"))
                        constr_at_time.append(ind.get("G"))

                #best_results.append(min(all_costs))
                all_results.append(cost_at_time)
                best_results.append(np.nanmin(np.array(cost_at_time), axis=0))
                all_constraints.append(constr_at_time)

        plt.style.use("seaborn-v0_8-paper")
        fig, ax = plt.subplots(dpi=1200)
        plt.gca().set_prop_cycle(color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0:2]) 
        ax.set_xlabel("Generation")
        ax.set_ylabel("EDP")
        platforms = ["Eyeriss", "Simba", "Gemmini"] #TODO

        # Iterate over each generation
        for idx, res in enumerate(all_results): 
            
            # By default, res has shape (n_indiv, n_obj). We want one scatter-plot per platform so we have to transpose
            results_of_generation = np.array(res).T
            for p, platform_res in enumerate(results_of_generation):
                ax.scatter(np.full_like(platform_res, idx, dtype=int).tolist(), platform_res, label=platforms[p])

        # Mark optimal results in each generation
        for idx, res in enumerate(best_results):
            best_results_of_generation = np.array(res)
            ax.scatter(np.full_like(best_results_of_generation, idx, dtype=int).tolist(), res,  label="Optimals", marker="+", color="r", linestyle="None")
        ax.plot(best_results, 'r+', label="Optimals")

        # Remove duplicate labels
        handles, labels = plt.gca().get_legend_handles_labels()
        dict_of_labels = dict(zip(labels, handles))
        plt.legend(dict_of_labels.values(), dict_of_labels.keys())

        os.makedirs(os.path.join(out_path, "dse_results", "figures"), exist_ok=True)
        out_file = os.path.join(out_path, "dse_results", "figures", "history.png")
        fig.savefig(out_file)

        # Dumpy all results of DSE history as npy file
        os.makedirs(os.path.join(out_path, "dse_results", "data"), exist_ok=True)
        npy_file_cost = os.path.join(out_path, "dse_results", "data", "history_data_cost.npy")
        with open(npy_file_cost, "wb") as f:
            np.save(f, np.array(all_results))

        npy_file_constr = os.path.join(out_path, "dse_results", "data", "history_data_constr.npy")
        with open(npy_file_constr, "wb") as f:
            np.save(f, np.array(all_constraints))
