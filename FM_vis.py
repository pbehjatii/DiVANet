def vis_FM(self, features, file_name):
  
  features = features.data.cpu().numpy()
  features = features.max(1, keepdims = True)
  features = features[0].transpose((1, 2, 0))
  features = features - features.min()
  features = features / features.max()
  features = features * 255
  features = features[:, :, 0]
  features = features.astype('uint8')
  
  plt.figure(figsize=(2, 2))
  ax = sns.heatmap(features, yticklabels=False, xticklabels=False)
  plt.savefig(filename + '.png')
  
  
  
