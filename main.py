from imports import *

#url = "http://www.cs.uu.nl/docs/vakken/mpr/data/mnist.csv"
data_set = pd.read_csv("mnist.csv")
mnist_data = data_set.values

# Get useful variables
labels = mnist_data[:, 0]
digits = mnist_data[:, 1:]
img_size = 28

# Gather descriptive statistics of the dataset
data_set.describe()

unique, counts = np.unique(labels, return_counts=True)
plt.bar(unique, counts)
plt.xticks(unique)
plt.title('Class Frequency')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.show()

###############################
#####       Step 1        #####
###############################
# Check the max and min values of the classes in the data set
labellist = [0] * 10
sum = 0
for x in range(len(labels)):
  labellist[labels[x]] += 1;
  sum += 1

print(labellist)
print("Most frequent class in data set: ", labellist.index(max(labellist)), " (", max(labellist),")\n",
      "Least frequent class in data set: ", labellist.index(min(labellist)), " (", min(labellist),")")

# Correct percentage if we would predict majority class:
print("Correct percentage if we would predict majority class:", 100 * max(labellist) / sum,"%")

# Avg and StDev
print("Average count for the labels:", st.mean(labellist))
print("Standard deviation of the labels:", st.stdev(labellist))

# Check which variables (features) are always 0
print(data_set.columns[(data_set == 0).all()])
def is_unique(s):
  a = s.to_numpy()
  return (a[0] == a).all()

useless_variables = []
for x in (data_set.columns[(data_set == 0).all()]):
  assert(data_set[x].sum(axis = 0) == 0)

for x in (data_set.columns):
  if (is_unique(data_set[x])):
    useless_variables.append(x)

###############################
#####       Step 2        #####
###############################
# Get each pixel color value and sum it up to calculate the "ink value" of the single digit
ink_values = []
for digit in digits:
  ink = 0
  for pixel in digit:
    ink += pixel
  ink_values.append(ink)

ink_vals_classes = []

ink = np.array([np.sum(row) for row in digits])
ink_mean = [np.mean(ink[labels == i]) for i in range(10)]
ink_std = [np.std(ink[labels == i]) for i in range(10)]

classes = np.unique(labels)
plt.bar(classes, ink_mean)
plt.plot(ink_std, color='yellow')
plt.xticks(classes)
plt.title('Ink Usage')
plt.xlabel('Class')
plt.ylabel('Ink Amount')
plt.show()

ink_scaled = scale(ink).reshape(-1, 1)

ink_model = LogisticRegression()
ink_model.fit(ink_scaled, labels)
ink_preds = ink_model.predict(ink_scaled)

ink_cm = confusion_matrix(labels, ink_preds)

#confusion_matrix = pd.crosstab(labels, ink_preds, rownames=['Actual'], colnames=['Predicted'])
plt.figure(figsize = (10,7))
#sn.heatmap(confusion_matrix, annot=True)

plt.figure(figsize = (10,7))
sn.heatmap(ink_cm, annot=True)
plt.show()


###############################
#####       Step 3        #####
###############################
# Count continuous pixels
pixels_count = []

for d in range(len(digits)):
  count = 0
  # print(digits[d])
  for p in range(len(digits[d])):
    if (digits[d][p] > 0):
      if (p+1 < len(digits[d]) and digits[d][p+1] > 0):
        count += 1
  pixels_count.append(count)

pixels_count = np.array(pixels_count)

pixels_mean = [np.mean(pixels_count[labels == i]) for i in range(10)]
pixels_std = [np.std(pixels_count[labels == i]) for i in range(10)]

classes = np.unique(labels)
plt.bar(classes, pixels_mean)
plt.plot(pixels_std, color='yellow')
plt.xticks(classes)
plt.title('Contiguous Pixels')
plt.xlabel('Class')
plt.ylabel('Number of Contiguous Pixels')
plt.show()

pixel_scaled = scale(pixels_count).reshape(-1, 1)

pixel_model = LogisticRegression()
pixel_model.fit(pixel_scaled, labels)
pixel_preds = pixel_model.predict(pixel_scaled)

pixel_cm = confusion_matrix(labels, pixel_preds)

plt.figure(figsize = (10,7))
sn.heatmap(pixel_cm, annot=True)
plt.show()

# Count pixels that aren't white
black = np.array([np.count_nonzero(row) for row in digits])
black_mean = [np.mean(black[labels == i]) for i in range(10)]
black_std = [np.std(black[labels == i]) for i in range(10)]

print("Length of first digit: ", len(digits[0]))
print("Non-white pixels in first digit: ", black[1])

classes = np.unique(labels)
plt.bar(classes, black_mean)
plt.plot(black_std, color='yellow')
plt.xticks(classes)
plt.title('Digit surface')
plt.xlabel('Class')
plt.ylabel('Non-white pixels')
plt.show()

black_scaled = scale(black).reshape(-1, 1)

black_model = LogisticRegression()
black_model.fit(black_scaled, labels)
black_preds = black_model.predict(black_scaled)

black_cm = confusion_matrix(labels, black_preds)

plt.figure(figsize = (10,7))
sn.heatmap(black_cm, annot=True)
plt.show()

# Count pixels that aren't white
white = np.array([np.count_nonzero(row) for row in digits])

for i in range(len(white)):
  white[i] = len(digits[i]) - white[i]

white_mean = [np.mean(white[labels == i]) for i in range(10)]
white_std = [np.std(white[labels == i]) for i in range(10)]

classes = np.unique(labels)
plt.bar(classes, white_mean)
plt.plot(white_std, color='yellow')
plt.xticks(classes)
plt.title('Digit surface')
plt.xlabel('Class')
plt.ylabel('White pixels')
plt.show()

white_scaled = scale(white).reshape(-1, 1)

white_model = LogisticRegression()
white_model.fit(white_scaled, labels)
white_preds = white_model.predict(white_scaled)

white_cm = confusion_matrix(labels, white_preds)

plt.figure(figsize = (10,7))
sn.heatmap(white_cm, annot=True)
plt.show()

# Count pixels that aren't white along X axis
x_counts = []
for d in digits:
  d = d.reshape(img_size, img_size)
  x_counts.append(np.count_nonzero(d, axis = 0))
x_counts = np.array(x_counts)

# Count pixels that aren't white along Y axis
y_counts = []
for d in digits:
  d = d.reshape(img_size, img_size)
  y_counts.append(np.count_nonzero(d, axis = 1))
y_counts = np.array(y_counts)

x_mean = [np.mean(x_counts[labels == i]) for i in range(10)]
x_std = [np.std(x_counts[labels == i]) for i in range(10)]

y_mean = [np.mean(y_counts[labels == i]) for i in range(10)]
y_std = [np.std(y_counts[labels == i]) for i in range(10)]

# Pixel count on X axis
classes = np.unique(labels)
plt.bar(classes, x_mean)
plt.plot(x_std, color='yellow')
plt.xticks(classes)
plt.title('Non-white pixels on X axis')
plt.xlabel('Class')
plt.ylabel('Number of non-white pixels')
plt.show()

# Pixel count on Y axis
classes = np.unique(labels)
plt.bar(classes, y_mean)
plt.plot(y_std, color='yellow')
plt.xticks(classes)
plt.title('Non-white pixels on Y axis')
plt.xlabel('Class')
plt.ylabel('Number of non-white pixels')
plt.show()

# Pixel count on X axis
digit_num = 5
pixels = np.arange(img_size)
xvalue = x_counts[digit_num]
yvalue = y_counts[digit_num]

print(xvalue)

# Plot pixel count along X axis
plt.figure(figsize=(10,10))
plt.bar(pixels, xvalue)
plt.xticks(pixels)
x_title = 'Non-white pixels on X axis for digit ' + str(digit_num)
plt.title(x_title)
plt.ylabel('Number of non-white pixels')
plt.show()

# Visualize first digit
fig = plt.figure(figsize = (10,10)) # create a 5 x 5 figure
ax = fig.add_subplot(111)
ax.set_xticks(np.arange(img_size))
ax.set_yticks(np.arange(img_size))
ax.imshow(digits[digit_num].reshape(img_size, img_size))
plt.show()

# Plot pixel count along Y axis
fig = plt.figure(figsize = (10,10)) # create a 5 x 5 figure
ax = fig.add_subplot(111)
ax.barh(pixels, yvalue)
ax.set_yticks(pixels)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Number of non-white pixels')
y_title = 'Non-white pixels on Y axis for digit ' + str(digit_num)
ax.set_title(y_title)
plt.show()

x_model = LogisticRegression()
x_model.fit(x_counts, labels)
x_preds = x_model.predict(x_counts)

x_cm = confusion_matrix(labels, x_preds)

plt.figure(figsize = (10,7))
sn.heatmap(x_cm, annot=True)
plt.show()

y_model = LogisticRegression()
y_model.fit(y_counts, labels)
y_preds = y_model.predict(y_counts)

y_cm = confusion_matrix(labels, y_preds)

plt.figure(figsize = (10,7))
sn.heatmap(y_cm, annot=True)
plt.show()

xy_counts = []
for i in range(len(digits)):
  xy_counts.append(np.concatenate((x_counts[i], y_counts[i])))
xy_counts = np.array(xy_counts)

print(x_counts[0])
print(y_counts[0])
print(xy_counts.shape)

xy_model = LogisticRegression()
xy_model.fit(xy_counts, labels)
xy_preds = xy_model.predict(xy_counts)

xy_cm = confusion_matrix(labels, xy_preds)

plt.figure(figsize = (10,7))
sn.heatmap(xy_cm, annot=True)
plt.show()