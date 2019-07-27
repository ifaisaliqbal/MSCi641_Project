import matplotlib.pyplot as plt



def plot_logistic():
	training_size = [40106, 36095, 32084, 28074, 24063, 20053, 16042, 12031, 8021, 4010, 2005]

	# Improved system
	logistic_regression_score_dev = [88.50, 88.20, 88.21, 87.79, 88.03, 88.01, 87.70, 86.75, 86.31, 82.87, 79.81]
	logistic_regression_score_test = [85.47, 85.21, 85.42, 85.09, 85.41, 85.57, 85.66, 85.33, 84.58, 82.61, 79.55]

	# Baseline system
	random_forest_score_dev = [64.14, 59.94, 59.59, 61.70, 59.85, 59.31, 53.85, 53.30, 48.51, 45.31, 43.14]
	random_forest_score_test = [65.96, 63.37, 60.16, 63.31, 58.39, 58.64, 55.05, 58.56, 54.32, 49.91, 44.87]

	plt.title("Learning curves of trainable systems")

	plt.plot(training_size[::-1], logistic_regression_score_dev[::-1], 'o-', color="r",)
	plt.plot(training_size[::-1], logistic_regression_score_test[::-1], 'o-', color="b",)
	plt.plot(training_size[::-1], random_forest_score_dev[::-1], 'o-', color="y",)
	plt.plot(training_size[::-1], random_forest_score_test[::-1], 'o-', color="g",)


	plt.legend(['Improved System on dev set','Improved System on test set', 'Baseline System on dev set', 'Baseline System on test set'], loc='lower right', fontsize=10)
	plt.xlabel('Training examples', fontsize=12)
	plt.ylabel('Accuracy score', fontsize=12)
	plt.grid()
	plt.show()


plot_logistic()