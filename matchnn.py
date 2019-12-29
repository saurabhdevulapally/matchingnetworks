import tensorflow as tf
from keras.layers.merge import _Merge

class MatchCosine(_Merge):
    
    def __init__(self,nway=5,**kwargs):
        super(MatchCosine,self).__init__(**kwargs)
        self.eps = 1e-10
        self.nway = nway

    def build(self, input_shape):
        if not isinstance(input_shape, list) or len(input_shape) != self.nway+2:
            raise ValueError('A ModelCosine layer should be called on a list of inputs of length %d'%(self.nway+2))

    def call(self,inputs):
        
        similarities = []

        targetembedding = inputs[-2] # embedding of the query image
        numsupportset = len(inputs)-2
        for ii in range(numsupportset):
            supportembedding = inputs[ii] # embedding for i^{th} member in the support set

            sum_support = tf.reduce_sum(tf.square(supportembedding), 1, keepdims=True)
            supportmagnitude = tf.math.rsqrt(tf.clip_by_value(sum_support, self.eps, float("inf"))) #reciprocal of the magnitude of the member of the support 

            sum_query = tf.reduce_sum(tf.square(targetembedding), 1, keepdims=True)
            querymagnitude = tf.math.rsqrt(tf.clip_by_value(sum_query, self.eps, float("inf"))) #reciprocal of the magnitude of the query image

            dot_product = tf.matmul(tf.expand_dims(targetembedding,1),tf.expand_dims(supportembedding,2))
            dot_product = tf.squeeze(dot_product,[1])

            cosine_similarity = dot_product*supportmagnitude*querymagnitude
            similarities.append(cosine_similarity)

        similarities = tf.concat(axis=1,values=similarities)
        softmax_similarities = tf.nn.softmax(similarities)
        preds = tf.squeeze(tf.matmul(tf.expand_dims(softmax_similarities,1),inputs[-1]))
        
        preds.set_shape((inputs[0].shape[0], self.nway))

        return preds

#changed the function as the original was erroneous
    def compute_output_shape(self,input_shape):
        input_shapes = input_shape
        return (self.nway, input_shapes[0][0])



