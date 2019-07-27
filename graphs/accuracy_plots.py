import matplotlib.pyplot as plt

width = 0.25


# IMPROVED_SYSTEM: Logistic, MNV, RF
development_score = [88.50, 57.32, 79.09]
test_score = [85.47, 58.80, 77.38]

# BASELINE_SYSTEM: Logistic, MNV, RF
#development_score = [60.10, 57.32, 64.14]
#test_score = [54.91, 54.84, 65.96]


# Plotting the bars
fig, ax = plt.subplots(figsize=(10,5))

pos = list(range(len(development_score)))

plt.bar(pos, development_score, width, alpha=0.5, color='#EE3224', label='Development set')
plt.bar([p + width for p in pos], test_score, width, alpha=0.5, color='#F78F1E', label='Test set')

# Set the y axis label
ax.set_ylabel('Accuracy score')

# Set the chart's title
ax.set_title('Accuracy scores for dev and test sets (improved system)')

# Set the position of the x ticks
ax.set_xticks([p + 1.5 * width for p in pos])

# Set the labels for the x ticks
ax.set_xticklabels(['Logistic regression', 'Multinomial Naive Bayes', 'Random Forest'])
ax.set_yticks([0, 20, 40, 60, 80, 100])


plt.legend(['Development Set', 'Test Set'], loc='upper left')


plt.xlim(min(pos)-width, max(pos)+width*4)
plt.ylim([0, max(development_score + test_score)*2])

plt.grid()
plt.show()