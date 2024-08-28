from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix

from model import MAX_SEQUENCE_LENGTH, get_model
from utills import get_class_weights, load_cui_dataset

DATASET_DIR = "data/cui/processed/"

train, dev, test = load_cui_dataset(DATASET_DIR, MAX_SEQUENCE_LENGTH)
model = get_model()

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])

print(model.summary())

checkpoints = ModelCheckpoint(
    "trained_models/model.{epoch:02d}-{val_loss:.3f}.weights.h5",
    monitor="acc",
    verbose=1,
    save_best_only=True,
    save_weights_only=True,
    mode="max",
)

model.fit(
    train[0],
    train[1],
    batch_size=500,
    epochs=5,
    shuffle=True,
    class_weight=get_class_weights(train[1]),
    validation_data=(dev[0], dev[1]),
    callbacks=[checkpoints],
)

print(model.evaluate(x=test[0], y=test[1], verbose=1))

y_pred = model.predict(test[0])
y_pred = y_pred > 0.5
y_pred = y_pred.astype(int)
y_true = test[1]
print(confusion_matrix(y_true, y_pred))
