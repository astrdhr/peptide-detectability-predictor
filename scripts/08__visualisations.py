

# Plot ROC curves


# Plot PR curves


# Plot histogram of detectability distributions


# Stacked-bar confusion matrix based on peptide length


# Accuracy bar chart for model generalisability


'''
Supplementary figures
'''

# Correlation matrix of AAIndex1 features
fig_dims = (12, 8)
fig, ax = plt.subplots(figsize=fig_dims)
sns.heatmap(X_train_aaindex1.corr(), ax=ax)
ax.set_title('Correlation matrix of calculated AAIndex1 features (train dataset)')
ax.set_xlabel('AAIndex1 Features (n=553)')
ax.set_xlabel('AAIndex1 Features (n=553)')
plt.show()
