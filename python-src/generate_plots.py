
import matplotlib.pyplot                as plt
import pathlib                          as path
import data_visualization.helpers       as helpers
import data_visualization.params        as cfg
import pprint                           as pp
import numpy                            as np
import caffe                            as caffe
import cv2                              as cv2
import seaborn                          as sn
import pandas                           as pd
import tabulate                         as tab
import json                             as json

from caffe_helpers.parse_log            import parse_log
from collections                        import Counter
from math                               import floor
from copy                               import deepcopy

figure_output_dir   = path.Path("../../report/figures")
log_file_dir        = path.Path("./training-logs/")

no_dropout = [[4187.0, 44.0, 8.0, 5.0, 2.0, 8.0, 16.0], [91.0, 1515.0, 58.0, 10.0, 0.0, 5.0, 15.0], [29.0, 177.0, 745.0, 69.0, 15.0, 3.0, 4.0], [11.0, 20.0, 158.0, 272.0, 78.0, 16.0, 9.0], [5.0, 1.0, 19.0, 102.0, 95.0, 49.0, 3.0], [5.0, 1.0, 5.0, 36.0, 66.0, 177.0, 3.0], [15.0, 13.0, 1.0, 1.0, 0.0, 5.0, 328.0]]
with_dropout = [[4210.0, 28.0, 5.0, 3.0, 2.0, 5.0, 17.0], [98.0, 1531.0, 50.0, 6.0, 0.0, 4.0, 5.0], [26.0, 122.0, 813.0, 73.0, 3.0, 3.0, 2.0], [13.0, 6.0, 114.0, 351.0, 64.0, 13.0, 3.0], [4.0, 0.0, 11.0, 84.0, 108.0, 66.0, 1.0], [5.0, 1.0, 2.0, 14.0, 50.0, 219.0, 2.0], [8.0, 5.0, 1.0, 0.0, 0.0, 0.0, 349.0]]
dropout_and_horizontal_flip = [[5017.0, 44.0, 5.0, 0.0, 0.0, 2.0, 11.0], [80.0, 1721.0, 56.0, 2.0, 0.0, 1.0, 11.0], [31.0, 114.0, 1053.0, 90.0, 7.0, 3.0, 4.0], [12.0, 7.0, 66.0, 479.0, 55.0, 12.0, 2.0], [6.0, 0.0, 9.0, 69.0, 140.0, 95.0, 1.0], [10.0, 1.0, 0.0, 15.0, 41.0, 314.0, 0.0], [6.0, 4.0, 1.0, 0.0, 0.0, 1.0, 402.0]]
opencfu_conf_mat = [[985, 464, 332, 167, 89, 98, 100], [286, 136, 118, 52, 34, 31, 21], [105, 51, 32, 16, 7, 13, 22], [36, 25, 23, 14, 4, 5, 4], [15, 6, 11, 5, 1, 2, 2], [40, 17, 17, 3, 2, 5, 1], [3612, 1172, 769, 376, 183, 227, 264]]
learning_rate_step = [[5033.0, 32.0, 5.0, 0.0, 0.0, 1.0, 8.0], [80.0, 1737.0, 44.0, 2.0, 0.0, 1.0, 7.0], [28.0, 135.0, 1069.0, 59.0, 4.0, 4.0, 3.0], [13.0, 5.0, 115.0, 439.0, 46.0, 13.0, 2.0], [6.0, 0.0, 13.0, 80.0, 149.0, 71.0, 1.0], [9.0, 1.0, 2.0, 15.0, 50.0, 303.0, 1.0], [6.0, 4.0, 3.0, 0.0, 0.0, 0.0, 401.0]]
more_dropout = [[5033.0, 33.0, 2.0, 1.0, 0.0, 0.0, 10.0], [78.0, 1743.0, 37.0, 1.0, 0.0, 0.0, 12.0], [36.0, 138.0, 1072.0, 50.0, 0.0, 2.0, 4.0], [16.0, 7.0, 104.0, 463.0, 33.0, 8.0, 2.0], [7.0, 0.0, 6.0, 88.0, 147.0, 68.0, 4.0], [10.0, 1.0, 2.0, 12.0, 42.0, 313.0, 1.0], [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 412.0]]
more_training_learning_rate_inv = [[5028.0, 42.0, 4.0, 0.0, 0.0, 1.0, 4.0], [71.0, 1739.0, 54.0, 2.0, 0.0, 1.0, 4.0], [31.0, 115.0, 1097.0, 54.0, 1.0, 1.0, 3.0], [12.0, 2.0, 113.0, 460.0, 42.0, 2.0, 2.0], [5.0, 0.0, 9.0, 77.0, 183.0, 46.0, 0.0], [10.0, 1.0, 2.0, 11.0, 53.0, 304.0, 0.0], [5.0, 7.0, 0.0, 0.0, 0.0, 0.0, 402.0]]
more_training_learning_rate_step = [[5025.0, 39.0, 7.0, 0.0, 0.0, 0.0, 8.0], [81.0, 1741.0, 40.0, 2.0, 0.0, 0.0, 7.0], [37.0, 123.0, 1090.0, 46.0, 1.0, 3.0, 2.0], [14.0, 6.0, 98.0, 473.0, 34.0, 6.0, 2.0], [6.0, 0.0, 7.0, 81.0, 146.0, 80.0, 0.0], [10.0, 2.0, 1.0, 17.0, 43.0, 308.0, 0.0], [9.0, 6.0, 1.0, 0.0, 0.0, 1.0, 397.0]]
undersampled = [[4984.0, 70.0, 10.0, 1.0, 0.0, 2.0, 12.0], [60.0, 1745.0, 53.0, 3.0, 0.0, 1.0, 9.0], [26.0, 128.0, 1099.0, 43.0, 1.0, 3.0, 2.0], [12.0, 4.0, 104.0, 469.0, 35.0, 7.0, 2.0], [4.0, 0.0, 9.0, 71.0, 159.0, 76.0, 1.0], [7.0, 1.0, 6.0, 12.0, 46.0, 309.0, 0.0], [8.0, 6.0, 0.0, 0.0, 0.0, 0.0, 400.0]]

def generate_learning_curve(training_data, validation_data, plot_name):

    # Extract the iteration count
    xs = [d_i["NumIters"] for d_i in training_data]
    ys = [d_i["loss"] for d_i in training_data]

    fig, ax1 = plt.subplots()
    first_line = helpers.plot_a_line(xs, ys, idx=0, label="training log loss",
            plot_markers=False, axis_to_plot_on=ax1)
    ax1.set_yticks([t_i for t_i in np.arange(0.0, 2.6, 0.5)])

    helpers.xlabel("Iteration", ax=ax1)
    helpers.ylabel("Training loss", ax=ax1)

    ax2 = ax1.twinx()
    helpers.ylabel(r"Validation accuracy (\%)", ax=ax2)
    xs = [d_i["NumIters"] for d_i in validation_data]
    ys = [d_i["accuracy"]*100 for d_i in validation_data]
    second_line = helpers.plot_a_line(xs, ys, idx=1, label="validation accuracy",
            plot_markers=False, axis_to_plot_on=ax2)
    ax2.set_ylim((82, 93))
    ax2.set_yticks([t_i for t_i in range(82, 94, 1)])

    ax1.legend(first_line + second_line, 
            ["training log loss", "validation accuracy"], ncol=2,
            **cfg.LEGEND)
    helpers.save_figure(plot_name, num_cols=2, no_legend=True)

def get_samples_labeled_as(conf_mat, idx):
    idx -= 1
    return conf_mat[idx]

def get_samples_that_are(conf_mat, idx):
    idx -= 1
    return [row[idx] for row in conf_mat]

def confusion_matrix(array, output_path, labels=["1", "2", "3", "4", "5", "6", "Outlier"]):
    # array = [[int(x_i)/sum for x_i in x] for x in array] 
    array = [[int(x_ij) for x_ij in x_i] for x_i in array]
    total_samples = sum([sum(x_i) for x_i in array])
    array = [[round(x_ij / sum(get_samples_that_are(array, actual_label+1)), 2) for actual_label, x_ij in enumerate(x_i)]
        for classified_as, x_i in enumerate(array)]
    print([sum(x_i) for x_i in array])
    df_cm = pd.DataFrame(array, index = labels,
                  columns = labels)
    plt.figure(figsize = (10,7))
    cmap = sn.color_palette("coolwarm", as_cmap=True)
    conf_mat = sn.heatmap(df_cm, annot=True, cmap=cmap)
    conf_mat.get_figure().savefig(output_path, bbox_inches="tight")
 
def make_precision_recall_table(conf_mat1, conf_mat2):

    # The confusion matrix is a 2d array where the i_th column of the j_th row indicates
    # the number of samples of class i that were labelled as class j by the algorithm.
    def compute_precision_recall_for(conf_mat):
        precision = []
        recall = []
        for class_label in range(1, 8):
            true_positives = get_samples_labeled_as(conf_mat, class_label)[class_label-1]
            true_and_false_positives = sum(get_samples_labeled_as(conf_mat, class_label))
            p = true_positives / true_and_false_positives
            true_positive_false_negative = sum(get_samples_that_are(conf_mat, class_label))
            r = true_positives / true_positive_false_negative
            precision.append(round(p, 2))
            recall.append(round(r, 2))
        return precision, recall, [compute_f1(*t) for t in zip(precision, recall)]

    def compute_f1(precision, recall):
        return round(2 * ((precision * recall) / (precision + recall)), 2)
    
    precision_1, recall_1, f_1 = compute_precision_recall_for(conf_mat1)
    precision_2, recall_2, f_2 = compute_precision_recall_for(conf_mat2)

    
    table = [[p_1, r_1, f_1, p_2, r_2, f_2] for p_1, r_1, f_1, p_2, r_2, f_2 in 
            zip(precision_1, recall_1, f_1, precision_2, recall_2, f_2)]
    table[0] = ["1 colony"] + table[0]
    for idx in range(1, len(table)-1):
        table[idx] = [f"{idx+1} colonies"] + table[idx]
    table[-1] = ["Outliers"] + table[-1]
    print(tab.tabulate(table, tablefmt="latex"))



def make_learning_curves():
    def _make_learning_curve(figure_output_path, log_path, num_iters=None):
        training_data, validation_data = parse_log(str(log_path))
        if num_iters != None:
            training_data = [d_i for d_i in training_data if d_i["NumIters"] < num_iters]
            validation_data = [d_i for d_i in validation_data if d_i["NumIters"] < num_iters]

        generate_learning_curve(training_data, validation_data, 
                figure_output_path)
    
    _make_learning_curve(figure_output_dir / "learning-curve-with-dropout.pdf",
            log_file_dir / "with-dropout.log")
    _make_learning_curve(figure_output_dir / "learning-curve-no-dropout.pdf",
            log_file_dir / "no-dropout.log")
    _make_learning_curve(figure_output_dir / "learning-curve-horizontal-flip.pdf",
            log_file_dir / "training-with-flip-dataset.log")
    _make_learning_curve(figure_output_dir / "learning-curve-lr-step.pdf",
            log_file_dir / "lr-step-training.log")
    _make_learning_curve(figure_output_dir / "learning-curve-more-dropout.pdf",
            log_file_dir / "dropout-0.75.log")
    _make_learning_curve(figure_output_dir / "learning-curve-more-iterations.pdf",
            log_file_dir / "concat.log")
    _make_learning_curve(figure_output_dir / "learning-curve-more-iterations-step.pdf",
            log_file_dir / "step-more-training.log")
    _make_learning_curve(figure_output_dir / "learning-curve-more-dropout-step.pdf",
            log_file_dir / "step-more-training.log", 50001)
    _make_learning_curve(figure_output_dir / "learning-curve-undersampled.pdf",
            log_file_dir / "undersampled.log", 50001)

def make_confusion_matrices():
    confusion_matrix(no_dropout, 
            figure_output_dir / "cnn-conf-mat-no-dropout.pdf")
    confusion_matrix(with_dropout,
            figure_output_dir / "cnn-conf-mat-with-dropout.pdf")
    confusion_matrix(dropout_and_horizontal_flip,
            figure_output_dir / "cnn-conf-mat-with-horizontal-flip.pdf")
    confusion_matrix(opencfu_conf_mat,
            figure_output_dir / "opencfu-conf-mat.pdf")
    confusion_matrix(learning_rate_step,
            figure_output_dir / "cnn-conf-mat-learning-rate-step.pdf")
    confusion_matrix(more_dropout,
            figure_output_dir / "cnn-conf-mat-more-dropout.pdf")
    confusion_matrix(more_training_learning_rate_inv ,
            figure_output_dir / "cnn-conf-mat-more-training-inv.pdf")
    confusion_matrix(more_training_learning_rate_step,
            figure_output_dir / "cnn-conf-mat-more-training-step.pdf")
    confusion_matrix(undersampled,
            figure_output_dir / "cnn-conf-mat-undersampled.pdf")

def dropout_vs_no_dropout_plot():
    with_dropout = log_file_dir / "with-dropout.log"
    no_dropout = log_file_dir / "no-dropout.log"
    more_dropout = log_file_dir / "dropout-0.75.log"

    _, dropout = parse_log(str(with_dropout))
    _, no_dropout = parse_log(str(no_dropout))
    _, more_dropout = parse_log(str(more_dropout))

    xs = [d_i["NumIters"] for d_i in dropout]
    dropout_ys = [100 * (1 - d_i["accuracy"]) for d_i in dropout]
    no_dropout_ys = [100 * (1 - d_i["accuracy"]) for d_i in no_dropout]
    more_dropout_ys = [100 * (1 - d_i["accuracy"]) for d_i in more_dropout]

    helpers.plot_a_line(xs, no_dropout_ys, label=r"$\mathbb{P}\{\text{dropout}\} = 0.0$", idx=3, plot_markers=False)
    helpers.plot_a_line(xs, dropout_ys, label=r"$\mathbb{P}\{\text{dropout}\} = 0.5$", idx=2, plot_markers=False)
    helpers.plot_a_line(xs, more_dropout_ys, label=r"$\mathbb{P}\{\text{dropout}\} = 0.75$", idx=4, plot_markers=False)
    helpers.ylim((5, 30))
    helpers.xlabel("Training Iteration")
    helpers.ylabel(r"Validation Error (\%)")

    helpers.save_figure(str(figure_output_dir / "dropout-comparison.pdf"), num_cols=2)

def augment_vs_no_augment():
    with_augment = log_file_dir / "training-with-flip-dataset.log"
    no_augment = log_file_dir / "with-dropout.log"

    augment_train, augment = parse_log(str(with_augment))
    no_augment_train, no_augment = parse_log(str(no_augment))

    xs = [d_i["NumIters"] for d_i in augment]
    augment_ys = [100 * (1 - d_i["accuracy"]) for d_i in augment]
    no_augment_ys = [100 * (1 - d_i["accuracy"]) for d_i in no_augment]

    helpers.plot_a_line(xs, augment_ys, label="Dataset augmentations", idx=4, plot_markers=False)
    helpers.plot_a_line(xs, no_augment_ys, label="Original dataset", idx=5, plot_markers=False)
    plt.ylim(7.5, 20.0)

    # augment_ys = [1 - d_i["loss"] for d_i in augment_train]
    # no_augment_ys = [1 - d_i["loss"] for d_i in no_augment_train]

    # xs = [d_i["NumIters"] for d_i in augment_train]
    # helpers.plot_a_line(xs, augment_ys, label="Dataset augmentations", idx=6, plot_markers=False)
    # helpers.plot_a_line(xs, no_augment_ys, label="Original dataset", idx=7, plot_markers=False)

    helpers.xlabel("Training Iteration")
    helpers.ylabel(r"Validation Error (\%)")
    helpers.save_figure(str(figure_output_dir / "dataset-augmentation-comparison.pdf"), num_cols=2)

def compute_opencfu_confusion_matrix(opencfu_results):
    conf_mat = [[0 for _ in range(7)] for _ in range(7)]

    print(sorted(Counter(r["label"] for r in opencfu_results.values()).items()))
    for result in opencfu_results.values():
        clamped = min(6, result["opencfu_prediction"])
        opencfu_predicted = result["opencfu_prediction"]
        # if opencfu predicts >6 colonies classify as outlier
        if opencfu_predicted > 6:
            opencfu_predicted = 6

        # if opencfu predicts 0 colonies, classify as outlier
        if opencfu_predicted == 0:
            opencfu_predicted == 6
        # for the labels, 0 => 1, 1 => 2 and so on

        opencfu_predicted -= 1 
        label = result["label"]
        conf_mat[opencfu_predicted][label] += 1

    return conf_mat

def make_all_precision_recall_tables():
    # make_precision_recall_table(no_dropout)
    # make_precision_recall_table(with_dropout)
    # make_precision_recall_table(opencfu_conf_mat)
    make_precision_recall_table(opencfu_conf_mat, dropout_and_horizontal_flip)

def generate_dataset_histogram(dataset, output_path):
    histogram = Counter([d_i["data"]["segment_type"]["data"] for d_i in dataset.values()])
    xs = [1, 2, 3, 4, 5, 6, 7]
    ys = [c[1]/sum(histogram.values()) for c in sorted(histogram.items())]
    plt.xticks(xs, [1, 2, 3, 4, 5, 6, "Outlier"])
    helpers.ylabel(r"$\mathbb{P}\{x = \mathcal{X}\}$")
    helpers.xlabel("Class Label")

    helpers.plot_a_bar(xs, ys, idx=1)
    helpers.save_figure(output_path, no_legend=True)

def generate_histograms():
    complete_dataset = json.loads(path.Path("./new-descriptor.json").read_text())
    undersampled_dataset = json.loads(path.Path("./segments-undersampled.json").read_text())

    generate_dataset_histogram(complete_dataset,
            figure_output_dir / "dataset-histogram.pdf")
    generate_dataset_histogram(undersampled_dataset, 
            figure_output_dir / "undersampled-dataset-histogram.pdf")

def generate_learning_rate_plots():
    # - step: return base_lr * gamma ^ (floor(iter / step))
    # - exp: return base_lr * gamma ^ iter
    # - inv: return base_lr * (1 + gamma * iter) ^ (- power)  
    base_lr = 0.01
    gamma_step = 0.9999
    gamma_inv = 0.0001
    step = 1
    power = 0.75
    xs = [x for x in range(1, 50001, 10)]
    inv_learning_rate = [base_lr * (1 + gamma_inv * iteration) ** (-power) for iteration in xs]
    step_learning_rate = [base_lr * (gamma_step**(floor(iteration/step))) for iteration in xs]

    helpers.plot_a_line(xs, step_learning_rate, label="Step", idx=6, plot_markers=False)
    helpers.plot_a_line(xs, inv_learning_rate, label="Inverse", idx=7, plot_markers=False)
    helpers.xlabel("Training Iteration")
    helpers.ylabel("Learning Rate")
    helpers.save_figure(figure_output_dir / "learning-rate.pdf", num_cols=2)

def generate_step_vs_inv_learning_rate():

    _, validation_data_step = parse_log(
            str(log_file_dir / "lr-step-training.log"))
    _, validation_data_inv = parse_log(
            str(log_file_dir / "training-with-flip-dataset.log"))

    xs = [d_i["NumIters"] for d_i in validation_data_step]

    helpers.plot_a_line(xs, ys_step, label="Step", plot_markers=False, idx=6)
    helpers.plot_a_line(xs, ys_inv, label="Inverse", plot_markers=False, idx=7)
    plt.ylim(7.5, 20)
    helpers.xlabel("Training Iterations")
    helpers.ylabel(r"Validation Error (\%)")

    helpers.save_figure(figure_output_dir / "learning-rate-comparison.pdf", num_cols=2)

def generate_learning_rate_comparison_plot():
    _, validation_data_step = parse_log(
            str(log_file_dir / "lr-step-training.log"))
    _, validation_data_inv = parse_log(
            str(log_file_dir / "training-with-flip-dataset.log"))

    fig, (plot1, plot2) = plt.subplots(2)

    xs = [d_i["NumIters"] for d_i in validation_data_step]
    ys_step = [100 * (1 - d_i["accuracy"]) for d_i in validation_data_step]
    ys_inv = [100 * (1 - d_i["accuracy"]) for d_i in validation_data_inv]
    plot2.set_ylim(7.5, 20)
    helpers.xlabel("Training Iterations", ax=plot2)
    helpers.ylabel(r"Validation Error (\%)", ax=plot2, formatter=lambda x: x)
    helpers.plot_a_line(xs, ys_inv, label="Step",
            plot_markers=False, idx=6, axis_to_plot_on=plot2)
    helpers.plot_a_line(xs, ys_step, label="Inverse",
            plot_markers=False, idx=7, axis_to_plot_on=plot2)

    base_lr = 0.01
    gamma_step = 0.9999
    gamma_inv = 0.0001
    step = 1
    power = 0.75
    xs = [x for x in range(1, 50001, 10)]
    inv_learning_rate = [base_lr * (1 + gamma_inv * iteration) ** (-power) 
            for iteration in xs]
    step_learning_rate = [base_lr * (gamma_step**(floor(iteration/step))) 
            for iteration in xs]
    helpers.plot_a_line(xs, step_learning_rate, idx=6, 
            plot_markers=False, axis_to_plot_on=plot1)
    helpers.plot_a_line(xs, inv_learning_rate, idx=7,
            plot_markers=False, axis_to_plot_on=plot1)
    plot1.xaxis.set_ticklabels([])
    plot1.grid(**cfg.GRID)
    helpers.ylabel("Learning Rate", ax=plot1, formatter=lambda x: x)

    legend_params = deepcopy(cfg.LEGEND)
    legend_params["bbox_to_anchor"] = (0.5, 0.975)
    fig.legend(ncol=2, **legend_params)
    helpers.save_figure(figure_output_dir / "learning-rate-comparison.pdf", 
            no_legend=True)

def generate_per_class_accuracy_bar_plot():
    conf_mat_1 = more_dropout
    conf_mat_2 = undersampled
    
    fig, (ax1, ax2) = plt.subplots(2)

    bar_width = 0.1
    bar_1_xs = np.arange(0.0, 0.3*6.5, 0.3)
    bar_2_xs = [x_i + bar_width for x_i in bar_1_xs]
    print(bar_1_xs, bar_2_xs)

    bar_1_ys = [100*(conf_mat_1[idx][idx] / sum(get_samples_that_are(conf_mat_1, idx+1))) 
            for idx in range(7)]
    bar_2_ys = [100*(conf_mat_2[idx][idx] / sum(get_samples_that_are(conf_mat_2, idx+1)))
            for idx in range(7)]

    plt.xticks([x_i + 0.5*bar_width for x_i in bar_1_xs], [1, 2, 3, 4, 5, 6, "Outlier"])
    helpers.plot_a_bar(bar_1_xs, bar_1_ys, idx=0, 
            label="Augmented Dataset", bar_width=bar_width, axis_to_plot_on=ax2)
    helpers.plot_a_bar(bar_2_xs, bar_2_ys, idx=1,
            label="Undersampled Dataset", bar_width=bar_width, axis_to_plot_on=ax2)
    helpers.xlabel("Class label")
    helpers.ylabel(r"Validation Accuracy (\%)", formatter=lambda x: x,
            ax=ax2)
    
    undersampled_dataset = json.loads(path.Path("./segments-undersampled.json")
            .read_text())
    histogram = Counter([d_i["data"]["segment_type"]["data"]
            for d_i in undersampled_dataset.values()])
    xs = [x_i for x_i in range(1, 8)]
    total_samples = sum(histogram.values())
    ys = [c[1] / total_samples for c in sorted(histogram.items())]
    helpers.plot_a_bar(xs, ys, idx=1, axis_to_plot_on = ax1,
            label_data=False)
    helpers.ylabel(r"$\mathbb{P}\{x = \mathcal{X}\}$",
            formatter=lambda x: x, ax=ax1)
    ax1.set_yticks([0.1, 0.2, 0.3])
    ax1.grid(**cfg.GRID)
    ax1.xaxis.set_ticklabels([])
    legend_params = deepcopy(cfg.LEGEND)
    legend_params["bbox_to_anchor"] = (0.5, 0.975)
    fig.legend(**legend_params, ncol=2)
    helpers.save_figure(figure_output_dir / "per-class-error-bar-plot.pdf",
            no_legend=True)


# So far I have results for three networks. The architecture of the networks is roughly the same 
# (pattern-recognition-2017) but there are some differences:
#   * no-dropout: No dropout in the fully connected layers
#   * with-dropout: dropout with p=0.5 in the fully connected layers
#   * horizontal-flip: Larger training set produced by augmenting the dataset with a horizontal flip

def generate_undersampling_plot():
    pass 

def main():
    # opencfu_results = json.loads(path.Path("./opencfu-results.json").read_text())
    # print(compute_opencfu_confusion_matrix(opencfu_results))
    # global opencfu_conf_mat
    # opencfu_conf_mat = compute_opencfu_confusion_matrix(opencfu_results)
    # pp.pprint(opencfu_conf_mat)
    # make_all_precision_recall_tables() 
    # make_learning_curves()
    # make_confusion_matrices()
    # dropout_vs_no_dropout_plot()
    # augment_vs_no_augment()
    # generate_histograms()
    # generate_learning_rate_plots()
    # generate_step_vs_inv_learning_rate()
    # generate_learning_rate_comparison_plot()
    generate_per_class_accuracy_bar_plot()

if __name__ == "__main__":
    main()
