import java.util.Date
import org.apache.log4j.LogManager
import org.apache.log4j.{Level, LogManager, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.Row
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{StructField, StructType}
import org.apache.spark.sql.{types, _}
import org.apache.spark.sql.SaveMode
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SQLContext
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.ml.feature.PCA
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.DataFrame
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.util.MLUtils
import java.io._

object analisis {

def getPCA(dataFrame: DataFrame, nc: Int): DataFrame = {
    val pca = new PCA()
    .setInputCol("features")
    .setOutputCol("pcaFeatures")
    .setK(nc)
    .fit(dataFrame)
    val pcaDF = pca.transform(dataFrame)
    return pcaDF }

//val PCs=getPCA(labeledDF,3)
//PCs.write.mode(SaveMode.Overwrite).saveAsTable("PCs")
def main(args: Array[String]) {
val logger = LogManager.getLogger("analisis")
logger.setLevel(Level.INFO)
logger.setLevel(Level.DEBUG)
Logger.getLogger("org").setLevel(Level.WARN)
Logger.getLogger("hive").setLevel(Level.WARN)
logger.info("Solicitando recursos a Spark")
val conf = new SparkConf().setAppName("AnalisisP2P_RF")
val sc = new SparkContext(conf)
val sqlContext = new org.apache.spark.sql.hive.HiveContext(sc)
var trees = Array(1, 5, 10, 20,50)
val numPartitions=25
logger.info("..........reading...............")
val data = (sqlContext.read
    .format("com.databricks.spark.csv")
    .option("header", "true")
    .option("inferSchema", "true")
    .load("data.csv")
    .drop(col("idsesion"))
    .drop(col("moneda"))
    .drop(col("tipo_tarjeta")))
val rows: RDD[Row] = data.rdd
val labeledPoints: RDD[LabeledPoint]=rows.map(row =>{LabeledPoint(row.getInt(16).toDouble,
 Vectors.dense(row.getDouble(0), row.getDouble(1),row.getDouble(2), row.getDouble(3),
 row.getDouble(4), row.getDouble(5),row.getDouble(6), row.getDouble(7),
 row.getDouble(8), row.getDouble(9),row.getDouble(10), row.getDouble(11),
 row.getDouble(12), row.getDouble(13),row.getDouble(14), row.getInt(15).toDouble))
 })
import sqlContext.implicits._
val labeledDF=labeledPoints.toDF()


val labelIndexer = (new StringIndexer()
.setInputCol("label")
.setOutputCol("indexedLabel")
.fit(labeledDF))

val featureIndexer = (new VectorIndexer()
.setInputCol("features")
.setOutputCol("indexedFeatures")
.setMaxCategories(4)
.fit(labeledDF))

var textOut="trees,tp,fn,fp,TPR,SPC,PPV,ACC,F1 \n"

for (x <- trees) {
val nTrees=x
val Array(trainingData, testData) = labeledDF.randomSplit(Array(0.7, 0.3))
val rf = (new RandomForestClassifier()
  .setLabelCol("indexedLabel")
  .setFeaturesCol("indexedFeatures")
  .setNumTrees(nTrees))

val labelConverter = (new IndexToString()
  .setInputCol("prediction")
  .setOutputCol("predictedLabel")
  .setLabels(labelIndexer.labels))

// Chain indexers and forest in a Pipeline
val pipeline = (new Pipeline()
  .setStages(Array(labelIndexer, featureIndexer, rf, labelConverter)))

logger.info("..........Training...............")
val model = pipeline.fit(trainingData)
val predictions = model.transform(testData)
logger.info("..........Calculate error...............")
val predRow: RDD[Row]=predictions.select("label", "predictedLabel").rdd
val predRDD: RDD[(Double, Double)] = (predRow.map(row=>
{(row.getDouble(0), row.getString(1).toDouble)}))
val tp=predRDD.filter(r=>r._1== 1.0 && r._2==1.0).count().toDouble
val fn=predRDD.filter(r=>r._1== 1.0 && r._2== -1.0).count().toDouble
val tn=predRDD.filter(r=>r._1== -1.0 && r._2== -1.0).count().toDouble
val fp=predRDD.filter(r=>r._1== -1.0 && r._2== 1.0).count().toDouble
val TPR = (tp/(tp+fn))*100.0
val SPC = (tn/(fp+tn))*100.0
val PPV= (tp/(tp+fp))*100.0
val acc= ((tp+tn)/(tp+fn+fp+tn))*100.0
val f1= ((2*tp)/(2*tp+fp+fn))*100.0
textOut=(textOut + tp + "," + fn + "," + tn + "," + fp + "," + TPR + "," + SPC + "," +
PPV + "," + acc + "," + f1  + "\n" )
println(textOut)
}

val pw = new PrintWriter(new File("Out.txt" ))
pw.write(textOut)
pw.close

//val evaluator = (new BinaryClassificationEvaluator()
//  .setLabelCol("indexedLabel"))

//val area = evaluator.evaluate(predictions)
//val rfModel = model.stages(2).asInstanceOf[RandomForestClassificationModel]
//rfModel.featureImportances


// "f1", "precision", "recall", "weightedPrecision", "weightedRecall"

//predictions.select("predictedLabel", "label", "features").show(10)

//val onlyF= labeledDF.filter("label= 1")
//val N=onlyF.count()
//val predictions = model.transform(onlyF).filter("predictedLabel=1").count()
//val rd=predictions/N.toDouble


/*
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

// [label: double, features: vector]
trainingData org.apache.spark.sql.DataFrame = ??? 
val nFolds: Int = ???
val NumTrees: Int = ???

val rf = new RandomForestClassifier()
  .setLabelCol("label")
  .setFeaturesCol("features")
  .setNumTrees(NumTrees)

val pipeline = new Pipeline().setStages(Array(rf)) 

val paramGrid = new ParamGridBuilder().build() // No parameter search

val evaluator = new MulticlassClassificationEvaluator()
  .setLabelCol("label")
  .setPredictionCol("prediction")
  // "f1", "precision", "recall", "weightedPrecision", "weightedRecall"
  .setMetricName("precision") 

val cv = new CrossValidator()
  // ml.Pipeline with ml.classification.RandomForestClassifier
  .setEstimator(pipeline)
  // ml.evaluation.MulticlassClassificationEvaluator
  .setEvaluator(evaluator) 
  .setEstimatorParamMaps(paramGrid)
  .setNumFolds(nFolds)

val model = cv.fit(trainingData) // trainingData: DataFrame

*/ 
     sc.stop()
  }
}


 