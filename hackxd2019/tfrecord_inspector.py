import tensorflow as tf

print "TRAIN.RECORD data ==>"
for example in tf.python_io.tf_record_iterator("data/train.record"):
    result = tf.train.Example.FromString(example)
    print result
    
print "TEST.RECORD data ==>"
for example in tf.python_io.tf_record_iterator("data/test.record"):
    result = tf.train.Example.FromString(example)
    print result
