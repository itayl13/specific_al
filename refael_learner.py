import pickle
from DataLoader.refael_data_loader import RefaelDataLoader
import os
from timed_active_learning import TimedActiveLearning, DistType
from ml_communities import MLCommunities, LearningMethod
import pandas as pd

RESULT_PKL_PATH = "results"


class RefaelLearner:
    def __init__(self):
        self._params = {
            'logger_name': "logger",
            # Data parameters
            'days_split': 1,
            'start_interval': 10,
            'database': 'Refael',
            'data_file_name': 'Refael_07_18.csv',  # should be in ../data/
            'date_format': "%Y-%m-%d",  # Refael
            'directed': True,
            'white_label': 1,
            # features + beta vectors parameters
            'max_connected': False,
            'ftr_pairs': 200,
            'identical_bar': 0.99,
            'context_beta': 1,
            # ML- parameters
            'learn_method': LearningMethod.XGBOOST,
            # AL - parameters
            'batch_size': 2,
            'queries_per_time': 30,
            'eps': 0.01,
            'target_recall': 0.7,
            'reveal_target': 0.6,
            'dist_type': DistType.Euclidian
        }
        self._database = RefaelDataLoader(os.path.join("data", self._params['data_file_name']), self._params)
        self._ml_learner = MLCommunities(method=self._params['learn_method'])
        # self._al_learner = ActiveLearning(self._params)

    def run_ml(self):
        # df = pd.DataFrame()
        time = 0
        while self._database.forward_time():
            print("-----------------------------------    TIME " + str(time) + "    ----------------------------------")
            time += 1
            beta_matrix, nodes_list, edges_list, labels = self._database.calc_curr_time()
            self._ml_learner.forward_time_data(beta_matrix, nodes_list, edges_list, labels)
        self._ml_learner.run()
            # df = pd.concat([df, self._ml_learner.run()])
        # if not os.path.exists('results'):
        #    os.mkdir('results')
        # writer = pd.ExcelWriter(os.path.join(os.getcwd(), 'results', 'time_run.xlsx'))
        # df.to_excel(writer, sheet_name='Sheet1', index=False)
        # writer.save()

    def run_al(self, pkl_result=False):
        if RESULT_PKL_PATH not in os.listdir("."):
            os.mkdir(RESULT_PKL_PATH)

        timed_al = TimedActiveLearning(self._params, self._database.num_blacks)
        # plot results y_axis
        recall = []
        # plot results x_axis
        revealed = []
        time = 0
        while self._database.forward_time():
            print("-----------------------------------    TIME " + str(time) + "    ----------------------------------")
            beta_matrix, nodes_list, edges_list, labels = self._database.calc_curr_time()
            rv, rec = timed_al.step(beta_matrix, labels)
            recall.append(rec)
            revealed.append(rv)
            time += 1

        if pkl_result:
            # save partial results to pkl
            pickle.dump([revealed, recall],
                        open(os.path.join(RESULT_PKL_PATH,
                             "al_results_split_" + str(int(self._params['days_split'])) +
                                          "_start_" + str(int(self._params['start_interval'])) +
                                          "_batch_" + str(int(self._params['batch_size'])) + ".pkl"), "wb"))
        return [revealed, recall]

    def run_al_simulation(self):
        if RESULT_PKL_PATH not in os.listdir("."):
            os.mkdir(RESULT_PKL_PATH)
        results = {}
        for split_interval in range(1, 3):
            for batch_size in range(1, 6):
                self._params['days_split'] = int(split_interval)
                self._params['batch_size'] = int(batch_size)
                self._params['start_interval'] = int(12 / split_interval)
                TOTAL_DAYS = 48
                TOTAL_COMMUNITIES = 586
                GOAL = 0.65
                total_time_intervals = TOTAL_DAYS/split_interval - self._params['start_interval'] + 1
                self._params['queries_per_time'] = round((GOAL * TOTAL_COMMUNITIES) /
                                                         (total_time_intervals * batch_size) + 0.5)
                self._database = RefaelDataLoader(os.path.join("data", self._params['data_file_name']),
                                                  self._params)
                results[(split_interval, batch_size)] = self.run_al()
        pickle.dump(results, open(os.path.join(RESULT_PKL_PATH, "simulation_results.pkl"), "wb"))


if __name__ == "__main__":
    r = RefaelLearner()
    # r.run_al_simulation()
    r.run_ml()
