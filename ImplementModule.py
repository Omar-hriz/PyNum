from TrainModel import *
import matplotlib.pyplot as plt

modelTest = MyModule()
modelTest.load_state_dict(torch.load("Model.pth"))

metric = 0
avg_loss = 0
for images, labels in train_data_loader:
    outputs = modelTest(images)
    avg_loss += F.cross_entropy(outputs, labels)
    metric += accuracy(outputs, labels)
print(metric / len(train_data_loader))
print(avg_loss/len(train_data_loader))

test_list = [test_dataset[0],test_dataset[123], test_dataset[69], test_dataset[193], test_dataset[1839], test_dataset[256]]
for image, label in test_list:
    print("image: {} predicted: {}".format(label, predictImage(image, modelTest)))
    plt.imshow(image[0], cmap="gray")
    plt.title("image: {} predicted: {}".format(label, predictImage(image, modelTest)))
    plt.show()