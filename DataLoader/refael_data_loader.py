from collections import Counter
from datetime import timedelta, datetime
import networkx as nx
import pandas as pd
from os import path
import pickle
import os

from beta_calculator import MotifRatio
from feature_calculators import FeatureMeta
from loggers import PrintLogger
from timed_graphs import TimedGraphs

from vertices.attractor_basin import AttractorBasinCalculator
from vertices.average_neighbor_degree import AverageNeighborDegreeCalculator
from vertices.betweenness_centrality import BetweennessCentralityCalculator
from vertices.bfs_moments import BfsMomentsCalculator
from vertices.closeness_centrality import ClosenessCentralityCalculator
from vertices.communicability_betweenness_centrality import CommunicabilityBetweennessCentralityCalculator
from vertices.eccentricity import EccentricityCalculator
from vertices.fiedler_vector import FiedlerVectorCalculator
from vertices.flow import FlowCalculator
from vertices.general import GeneralCalculator
from vertices.k_core import KCoreCalculator
from vertices.load_centrality import LoadCentralityCalculator
from vertices.louvain import LouvainCalculator
from vertices.motifs import nth_nodes_motif

MOTIF_FEATURES = {
    "motif3": FeatureMeta(nth_nodes_motif(3), {"m3"}),
    "motif4": FeatureMeta(nth_nodes_motif(4), {"m4"}),
}

ALL_FEATURES = {
    "attractor_basin": FeatureMeta(AttractorBasinCalculator, {"ab"}),  # directed only
    "average_neighbor_degree": FeatureMeta(AverageNeighborDegreeCalculator, {"avg_nd"}),
    "betweenness_centrality": FeatureMeta(BetweennessCentralityCalculator, {"betweenness"}),
    "bfs_moments": FeatureMeta(BfsMomentsCalculator, {"bfs"}),
    "closeness_centrality": FeatureMeta(ClosenessCentralityCalculator, {"closeness"}),
    "communicability_betweenness_centrality": FeatureMeta(CommunicabilityBetweennessCentralityCalculator,
                                                          {"communicability"}),
    "eccentricity": FeatureMeta(EccentricityCalculator, {"ecc"}),
    "fiedler_vector": FeatureMeta(FiedlerVectorCalculator, {"fv"}),
    "flow": FeatureMeta(FlowCalculator, {}),
    "general": FeatureMeta(GeneralCalculator, {"gen"}),
    # Isn't OK - also in previous version
    # "hierarchy_energy": FeatureMeta(HierarchyEnergyCalculator, {"hierarchy"}),
    "k_core": FeatureMeta(KCoreCalculator, {"kc"}),
    "load_centrality": FeatureMeta(LoadCentralityCalculator, {"load_c"}),
    "louvain": FeatureMeta(LouvainCalculator, {"lov"}),
    "motif3": FeatureMeta(nth_nodes_motif(3), {"m3"}),
    # "multi_dimensional_scaling": FeatureMeta(MultiDimensionalScalingCalculator, {"mds"}),
    # "page_rank": FeatureMeta(PageRankCalculator, {"pr"}),
    "motif4": FeatureMeta(nth_nodes_motif(4), {"m4"}),
    # "first_neighbor_histogram": FeatureMeta(nth_neighbor_calculator(1), {"fnh", "first_neighbor"}),
    # "second_neighbor_histogram": FeatureMeta(nth_neighbor_calculator(2), {"snh", "second_neighbor"}),
}

SOURCE = 'SourceID'
DEST = 'DestinationID'
DURATION = 'Duration'
TIME = 'StartTime'
COMMUNITY = 'Community'
TARGET = 'target'

ALL_BETA_PATH = "all_times_beta"
DATA_TARGET_FOLDER = "data_by_time"


class RefaelDataLoader:
    def __init__(self, data_path, params):
        # parameters - dictionary must contain { database: , logger_name: , date_format: , directed :
        # , max_connected : , ftr_pairs : , identical_bar : , context_beta: }
        self._params = params
        self._white = params['white_label']
        # number of days represented by one time interval
        self._time_split = self._params['days_split']
        self._all_beta_path = ALL_BETA_PATH + "_split_" + str(self._time_split) + ".pkl"
        self._start_interval = self._params['start_interval']
        # where to save splitted graph
        self._target_path = os.path.join(DATA_TARGET_FOLDER, params['database'], "split_" + str(self._time_split))
        self._logger = PrintLogger(self._params['logger_name'])
        self._params['files_path'] = self._target_path
        self._data_path = data_path
        # split to time intervals - only
        self._partition_data()
        self._timed_graph = None

        self.calc_all_times()       # calc all features for all times and save as pickle
        self._time_idx = 0
        # TOTAL NUMBER OF BLACK IN FINAL TIME
        self.num_blacks = sum([val for key, val in Counter(self._all_times_data[- 1][0][3]).items()
                               if key != self._white])

    # load from pkl or calculate and dump
    def calc_all_times(self):
        if os.path.exists(self._all_beta_path):
            self._all_times_data = pickle.load(open(self._all_beta_path, "rb"))
            return
        self._timed_graph = self._init_timed_graph()
        self._all_times_data = []
        # loop over all time intervals
        while self._forward_time():
            # calc features and beta for each time
            self._all_times_data.append([self._calc_curr_time()])
        pickle.dump(self._all_times_data, open(self._all_beta_path, "wb"))

    # init timed graph to time_0 - without calculating features/ beta-vectors
    def _init_timed_graph(self):
        return TimedGraphs(self._params['database'], start_time=self._start_interval,
                           files_path=self._params['files_path'],
                           logger=self._logger, features_meta=ALL_FEATURES,
                           directed=self._params['directed'], date_format=self._params['date_format'],
                           largest_cc=self._params['max_connected'])

    # split data to time intervals
    def _partition_data(self):
        # make target dir
        if DATA_TARGET_FOLDER not in os.listdir("."):
            os.mkdir(DATA_TARGET_FOLDER)
        if self._params['database'] not in os.listdir(DATA_TARGET_FOLDER):
            os.mkdir(os.path.join(DATA_TARGET_FOLDER, self._params['database']))
        if "split_" + str(self._time_split) not in \
                os.listdir(os.path.join(DATA_TARGET_FOLDER, self._params['database'])):
            os.mkdir(os.path.join(DATA_TARGET_FOLDER, self._params['database'], "split_" + str(self._time_split)))

        # open basic data csv (with all edges of all times)
        data_df = pd.read_csv(self._data_path)
        self._format_data(data_df)
        # make TIME column the index column and sort data by it
        data_df = data_df.set_index(TIME).sort_index()

        # time delta equals number of days for one time interval
        one_day = timedelta(days=self._time_split)
        # first day is the first row
        curr_day = data_df.first_valid_index()
        file_time = open(path.join(self._target_path, str(curr_day.date())), "wt")
        # next day is the  floor(<current_day>) + <one_time_interval>
        next_day = curr_day - timedelta(hours=curr_day.hour, minutes=curr_day.minute, seconds=curr_day.second) + one_day
        for curr_day, row in data_df.iterrows():
            # if curr day is in next time interval
            if curr_day >= next_day:
                # close and open new file
                file_time.close()
                next_day = curr_day - timedelta(hours=curr_day.hour, minutes=curr_day.minute,
                                                seconds=curr_day.second) + one_day
                file_time = open(path.join(self._target_path, str(curr_day.date())), "wt")

            # write edge to file
            file_time.write(str(row[SOURCE]) + " " + str(row[DEST]) + " " + str(row[DURATION]) + " "
                            + str(row[COMMUNITY]) + " " + str(row[TARGET]) + "\n")

    @staticmethod
    def _format_data(graph_df):
        graph_df[TIME] = graph_df[TIME]/1000                                           # milliseconds to seconds
        graph_df[TIME] = graph_df[TIME].apply(lambda x: datetime.fromtimestamp(x))     # to datetime format

    def _forward_time(self):
        return self._timed_graph.forward_time()

    def _calc_curr_time(self):
        # pick best features and calculate beta vectors
        beta = MotifRatio(self._timed_graph, is_directed=self._params['directed'])
        return beta.beta_matrix(), self._timed_graph.nodes_count_list(), \
               self._timed_graph.edges_count_list(), self._timed_graph.get_labels()

    # function for outside user they are only returning pre-calculated beta/labels
    def calc_curr_time(self):
        return self._all_times_data[self._time_idx - 1][0]

    def forward_time(self):
        if self._time_idx == len(self._all_times_data):
            return False
        self._time_idx += 1
        return True
