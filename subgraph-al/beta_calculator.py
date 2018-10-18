import os
import pickle

from sklearn.decomposition import PCA

from graph_features import GraphFeatures
from loggers import BaseLogger, PrintLogger
import numpy as np
from scipy.misc import comb

from timed_graphs import TimedGraphs


MOTIFS_VAR_PATH = os.path.join(os.sep, os.getcwd(), "graph-measures", "features_algorithms")


class BetaCalculator:
    def __init__(self, graphs: TimedGraphs, feature_pairs=None, logger: BaseLogger=None):
        if logger:
            self._logger = logger
        else:
            self._logger = PrintLogger("default graphs logger")
        self._graphs = graphs
        self._ftr_pairs = feature_pairs
        num_features = graphs.features_matrix(0).shape[1]
        num_rows = len(feature_pairs) if feature_pairs else int(comb(num_features, 2))
        self._beta_matrix = np.zeros((self._graphs.number_of_graphs(), num_rows))
        self._build()

    def _build(self):
        graph_index = 0
        for g_id in self._graphs.graph_names():
            self._logger.debug("calculating beta vec for:\t" + g_id)
            self._beta_matrix[graph_index, :] = self._calc_beta(g_id)
            graph_index += 1

    def _calc_beta(self, gid):
        raise NotImplementedError()

    def beta_matrix(self):
        return self._beta_matrix

    def to_file(self, file_name):
        out_file = open(file_name, "rw")
        for i in range(self._graphs.number_of_graphs()):
            out_file.write(self._graphs.index_to_name(i))  # graph_name
            for j in range(len(self._ftr_pairs)):
                out_file.write(str(self._beta_matrix[i][j]))  # beta_vector
            out_file.write("\n")
        out_file.close()


class MotifRatio:
    def __init__(self, graphs: TimedGraphs, is_directed, pca_size=20, logger: BaseLogger=None):
        self._pca_n_component = pca_size
        self._is_directed = is_directed                 # are the graphs directed
        self._index_ftr = None                          # list of ftr names + counter [ ... (ftr_i, 0), (ftr_i, 1) ...]
        self._beta_matrix = None                        # matrix of vectors for all graphs
        self._logger = logger if logger else PrintLogger("graphs logger")
        # self._graph_order = graph_order if graph_order else [g for g in sorted(graph_ftr_dict)]
        self._graphs = graphs
        # list index in motif to number of edges in the motif
        # self._motif_index_to_edge_num = {"motif3": self._motif_num_to_number_of_edges(3),
        #                                  "motif4": self._motif_num_to_number_of_edges(4)}
        self._build()

    def _build(self):
        # get vector for each graph and stack theme
        self._beta_matrix = np.vstack([self._feature_vector_(g_id) for g_id in self._graphs.graph_names()])
        # TODO -- PCA + add node and edges data -- not working so good for now
        # pca = PCA(n_components=self._pca_n_component)
        # self._beta_matrix = pca.fit_transform(self._beta_matrix)
        # self._beta_matrix = np.concatenate((self._beta_matrix, np.matrix(self._graphs.nodes_count_list()).T), axis=1)
        # self._beta_matrix = np.concatenate((self._beta_matrix, np.matrix(self._graphs.edges_count_list()).T), axis=1)

    def beta_matrix(self):
        return self._beta_matrix

    # load motif variation file
    def _load_variations_file(self, level):
        fname = "%d_%sdirected.pkl" % (level, "" if self._is_directed else "un")
        fpath = os.path.join(MOTIFS_VAR_PATH, "motif_variations", fname)
        return pickle.load(open(fpath, "rb"))

    # return dictionary { motif_index: number_of_edges }
    def _motif_num_to_number_of_edges(self, level):
        motif_edge_num_dict = {}
        for bit_sec, motif_num in self._load_variations_file(level).items():
            motif_edge_num_dict[motif_num] = bin(bit_sec).count('1')
        return motif_edge_num_dict

    # map matrix rows to features + count if there's more then one from feature
    def _set_index_to_ftr(self, gnx, gnx_ftr):
        if not self._index_ftr:
            sorted_ftr = [f for f in sorted(gnx_ftr) if gnx_ftr[f].is_relevant()]  # fix feature order (names)
            self._index_ftr = []
            temp_node = [x for x in gnx.nodes()][0]                                # pick arbitrary node
            for ftr in sorted_ftr:
                temp = gnx_ftr[ftr].feature(temp_node).tolist()
                temp = temp if type(temp) is list else [temp]                      # feature vector for a node
                # fill list with (ftr, counter)
                self._index_ftr += self._get_motif_type(ftr, len(temp)) if ftr == 'motif3' or ftr == 'motif4' else \
                    [(ftr, i) for i in range(len(temp))]

    # get feature vector for a graph
    def _feature_vector(self, gid):
        # get gnx gnx
        gnx, gnx_ftr = self._graphs.features_matrix(gid)
        ftr_mx = gnx_ftr.to_matrix(dtype=np.float32, mtype=np.matrix, should_zscore=False)
        final_vec = np.zeros((1, ftr_mx.shape[1]))

        self._set_index_to_ftr(gnx, gnx_ftr)

        motif3_ratio = None
        motif4_ratio = None
        for i, (ftr, ftr_count) in enumerate(self._index_ftr):
            if ftr == "motif3":
                # calculate { motif_index: motif ratio }
                motif3_ratio = self._count_subgraph_motif_by_size(ftr_mx, ftr) if not motif3_ratio else motif3_ratio
                final_vec[0, i] = motif3_ratio[ftr_count]
            elif ftr == "motif4":
                # calculate { motif_index: motif ratio }
                motif4_ratio = self._count_subgraph_motif_by_size(ftr_mx, ftr) if not motif4_ratio else motif4_ratio
                final_vec[0, i] = motif4_ratio[ftr_count]
            else:
                # calculate average of column
                final_vec[0, i] = np.sum(ftr_mx[:, i]) / ftr_mx.shape[0]
        return final_vec

    def _feature_vector_(self, gid):
        gnx, gnx_ftr = self._graphs.features_matrix(gid)
        ftr_mx = gnx_ftr.to_matrix(dtype=np.float32, mtype=np.matrix, should_zscore=True)
        final_vec = np.zeros((1, ftr_mx.shape[1]))

        self._set_index_to_ftr(gnx, gnx_ftr)

        for i in enumerate(self._index_ftr):
                final_vec[0, i[0]] = np.sum(ftr_mx[:, i[0]]) / ftr_mx.shape[0]
        return final_vec

    # return { motif_index: sum motif in index/ total motifs with same edge count }
    def _count_subgraph_motif_by_size(self, ftr_mat, motif_type):
        sum_dict = {ftr_count: np.sum(ftr_mat[:, i]) for i, (ftr, ftr_count) in enumerate(self._index_ftr)
                    if ftr == motif_type}       # dictionary { motif_index: sum column }
        sum_by_edge = {}                        # dictionary { num_edges_in_motif: sum of  }
        for motif_count, sum_motif in sum_dict.items():
            key = self._motif_index_to_edge_num[motif_type][motif_count]  # Don't forget self._motif_index_to_edge_num
            sum_by_edge[key] = sum_by_edge.get(key, 0) + sum_motif
        # rewrite dictionary { motif_index: sum column/ total motifs with same edge count }
        for motif_count in sum_dict:
            key = self._motif_index_to_edge_num[motif_type][motif_count]
            sum_dict[motif_count] = sum_dict[motif_count] / sum_by_edge[key] if sum_by_edge[key] else 0
        return sum_dict

    # return [ ... (motif_type, counter) ... ]
    def _get_motif_type(self, motif_type, num_motifs):
        header = []
        for i in range(num_motifs):
            header.append((motif_type, i))
        return header

    @staticmethod
    def is_motif(ftr):
        return ftr == 'motif4' or ftr == "motif3"



