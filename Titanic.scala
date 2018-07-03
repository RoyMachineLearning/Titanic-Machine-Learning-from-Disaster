// Databricks notebook source
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.sql.{SQLContext, Row, DataFrame, Column}

import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.{Pipeline, PipelineModel, Transformer}
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}

import org.apache.spark.ml.feature.{Imputer, ImputerModel}
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, IndexToString} 
import org.apache.spark.ml.feature.{VectorAssembler, VectorIndexer}
import org.apache.spark.ml.feature.{Bucketizer,Normalizer}

import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.classification._
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel,DecisionTreeClassifier}
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.ml.classification.NaiveBayes
import scala.collection.mutable
import com.microsoft.ml.spark.LightGBMClassifier
import ml.dmlc.xgboost4j.scala.spark.{XGBoostEstimator, XGBoostClassificationModel}

// COMMAND ----------

// Setup the Scala syntax for Spark SQL
val sparkImplicits = spark
import sparkImplicits.implicits._

// COMMAND ----------

val trainingDF = (sqlContext.read
                  .option("header","true")
                  .option("inferSchema","true")
                  .format("csv")
                  .load("/FileStore/tables/train.csv"))
  
val testDF = (sqlContext.read
                 .option("header","true")
                 .option("inferSchema","true")
                 .format("csv")
                 .load("/FileStore/tables/test.csv"))


// COMMAND ----------

trainingDF.printSchema()

// COMMAND ----------

val trainFeatures = (trainingDF
    .withColumn("Surname", regexp_extract($"Name","([\\w ']+),",1))
    .withColumn("Salutation", regexp_extract($"Name","(.*?)([\\w]+?)[.]",2))
    .withColumn("Mil", when($"Salutation" === "Col" or 
        $"Salutation" === "Major" or 
        $"Salutation" === "Capt", 1).otherwise(0))
    .withColumn("Doc", when($"Salutation" === "Dr", 1).otherwise(0))
    .withColumn("Rev", when($"Salutation" === "Rev", 1).otherwise(0))
    .withColumn("Nob", when($"Salutation" === "Sir" or
        $"Salutation" === "Countess" or
        $"Salutation" === "Count" or
        $"Salutation" === "Duke" or
        $"Salutation" === "Duchess" or
        $"Salutation" === "Jonkheer" or
        $"Salutation" === "Don" or
        $"Salutation" === "Dona" or
        $"Salutation" === "Lord" or
        $"Salutation" === "Lady" or
        $"Salutation" === "Earl" or
        $"Salutation" === "Baron", 1).otherwise(0))
    .withColumn("Mr", when($"Salutation" === "Mr", 1).otherwise(0))
    .withColumn("Mrs", when($"Salutation" === "Mrs" or 
        $"Salutation" === "Mme", 1).otherwise(0))
    .withColumn("Miss", when($"Salutation" === "Miss" or $"Salutation" === "Ms" or
        $"Salutation" === "Mlle", 1).otherwise(0))
    .withColumn("Mstr", when($"Salutation" === "Master", 1).otherwise(0))
    .withColumn("TotalFamSize",$"SibSp"+$"Parch"+1)
    .withColumn("Singleton", when($"TotalFamSize" === 1, 1).otherwise(0))
    .withColumn("SmallFam", when($"TotalFamSize" <= 4 && 
        $"TotalFamSize" > 1, 1).otherwise(0))
    .withColumn("LargeFam", when($"TotalFamSize" >= 5, 1).otherwise(0))
    .withColumn("Child", when($"Age" <= 18, 1).otherwise(0))
    .withColumn("Mother", when($"Age" > 15 && 
        $"Parch" > 0 && 
        $"Miss" === 0 && 
        $"Sex" === "female",1).otherwise(0)))

// COMMAND ----------

val testFeatures = (testDF
    .withColumn("Surname", regexp_extract($"Name","([\\w ']+),",1))
    .withColumn("Salutation", regexp_extract($"Name","(.*?)([\\w]+?)[.]",2))
    .withColumn("Mil", when($"Salutation" === "Col" or 
        $"Salutation" === "Major" or 
        $"Salutation" === "Capt", 1).otherwise(0))
    .withColumn("Doc", when($"Salutation" === "Dr", 1).otherwise(0))
    .withColumn("Rev", when($"Salutation" === "Rev", 1).otherwise(0))
    .withColumn("Nob", when($"Salutation" === "Sir" or
        $"Salutation" === "Countess" or
        $"Salutation" === "Count" or
        $"Salutation" === "Duke" or
        $"Salutation" === "Duchess" or
        $"Salutation" === "Jonkheer" or
        $"Salutation" === "Don" or
        $"Salutation" === "Dona" or
        $"Salutation" === "Lord" or
        $"Salutation" === "Lady" or
        $"Salutation" === "Earl" or
        $"Salutation" === "Baron", 1).otherwise(0))
    .withColumn("Mr", when($"Salutation" === "Mr", 1).otherwise(0))
    .withColumn("Mrs", when($"Salutation" === "Mrs" or 
        $"Salutation" === "Mme", 1).otherwise(0))
    .withColumn("Miss", when($"Salutation" === "Miss" or $"Salutation" === "Ms" or
        $"Salutation" === "Mlle", 1).otherwise(0))
    .withColumn("Mstr", when($"Salutation" === "Master", 1).otherwise(0))
    .withColumn("TotalFamSize",$"SibSp"+$"Parch"+1)
    .withColumn("Singleton", when($"TotalFamSize" === 1, 1).otherwise(0))
    .withColumn("SmallFam", when($"TotalFamSize" <= 4 && 
        $"TotalFamSize" > 1, 1).otherwise(0))
    .withColumn("LargeFam", when($"TotalFamSize" >= 5, 1).otherwise(0))
    .withColumn("Child", when($"Age" <= 18, 1).otherwise(0))
    .withColumn("Mother", when($"Age" > 15 && 
        $"Parch" > 0 && 
        $"Miss" === 0 && 
        $"Sex" === "female",1).otherwise(0)))

// COMMAND ----------

// convert the features manipulation to a temp view
trainFeatures.createOrReplaceTempView("trainFeatures")
testFeatures.createOrReplaceTempView("testFeatures")

// Explore the data
(trainFeatures
  .groupBy("Pclass","Embarked")
  .agg(count("*"),avg("Fare"),min("Fare"),max("Fare"),stddev("Fare"))
  .orderBy("Pclass","Embarked")
  .show())

// COMMAND ----------

/*Fill the missing value for Embarked and fare*/
val trainEmbarked = trainFeatures.na.fill("C",Seq("Embarked"))
val testFeaturesNew = testFeatures.na.fill(10.5,Seq("Fare"))
val testEmbarked = testFeaturesNew.na.fill("C", Seq("Embarked"))

// COMMAND ----------

// convert the features manipulation to a temp view
trainEmbarked.createOrReplaceTempView("trainEmbarked")
testEmbarked.createOrReplaceTempView("testEmbarked")

// COMMAND ----------

display(testEmbarked)

// COMMAND ----------

// MAGIC %sql SELECT Salutation FROM testEmbarked WHERE Embarked IS NULL

// COMMAND ----------

// MAGIC %sql SELECT Salutation FROM testEmbarked WHERE Fare IS NULL

// COMMAND ----------

// MAGIC %sql SELECT Salutation,percentile_approx(fare, 0.5) as median_fare FROM testEmbarked WHERE fare IS NOT NULL GROUP BY Salutation

// COMMAND ----------

// MAGIC %sql SELECT Salutation,count(*) as nullAge FROM trainEmbarked WHERE Age IS NULL GROUP BY Salutation

// COMMAND ----------

// MAGIC %sql SELECT Salutation,percentile_approx(Age, 0.5) AS Median_Age FROM trainEmbarked WHERE Age IS NOT NULL AND Salutation IN ('Miss','Master','Mr','Dr','Mrs') GROUP BY Salutation

// COMMAND ----------

// MAGIC %sql SELECT Salutation,count(*) as nullAge FROM testEmbarked WHERE Age IS NULL GROUP BY Salutation

// COMMAND ----------

// MAGIC %sql SELECT Salutation,percentile_approx(Age, 0.5) AS Median_Age FROM testEmbarked WHERE Age IS NOT NULL AND Salutation IN ('Miss','Master','Mr','Ms','Mrs') GROUP BY Salutation

// COMMAND ----------

// Impute the missing Age values for the relevant saluation columns and union the data back together for Training set
val trainMissDF = trainEmbarked.na.fill(21.0,Seq("Age")).where("Salutation = 'Miss'")
val trainMasterDF = trainEmbarked.na.fill(3.0,Seq("Age")).where("Salutation = 'Master'")
val trainMrDF = trainEmbarked.na.fill(30.0,Seq("Age")).where("Salutation = 'Mr'")
val trainDrDF = trainEmbarked.na.fill(44.0,Seq("Age")).where("Salutation = 'Dr'")
val trainMrsDF = trainEmbarked.na.fill(35.0,Seq("Age")).where("Salutation = 'Mrs'")

// Impute the missing Age values for the relevant saluation columns and union the data back together for Test Set
val testMrDF = testEmbarked.na.fill(28.5,Seq("Age")).where("Salutation = 'Mr'")
val testMissDF = testEmbarked.na.fill(22,Seq("Age")).where("Salutation = 'Miss'")
val testMasterDF = testEmbarked.na.fill(7,Seq("Age")).where("Salutation = 'Master'")
val testMsDF = testEmbarked.na.fill(36,Seq("Age")).where("Salutation = 'Ms'")
val testMrsDF = testEmbarked.na.fill(36,Seq("Age")).where("Salutation = 'Mrs'")

// COMMAND ----------

val trainRemainderDF = spark.sql("SELECT * FROM trainEmbarked WHERE Salutation NOT IN ('Miss','Master','Mr','Dr', 'Mrs')")
val trainCombinedDF = trainRemainderDF.union(trainMissDF).union(trainMasterDF).union(trainMrDF).union(trainDrDF).union(trainMrsDF)

// COMMAND ----------

val testRemainderDF = spark.sql("SELECT * FROM testEmbarked WHERE Salutation NOT IN ('Mr', 'Miss', 'Master','Ms','Mrs')")
val testCombinedDF = testRemainderDF.union(testMrDF).union(testMissDF).union(testMasterDF).union(testMsDF).union(testMrsDF)

// COMMAND ----------

display(testCombinedDF)

// COMMAND ----------

// Convert the categorical (string) values into numeric values
val genderIndexer = new StringIndexer().setInputCol("Sex").setOutputCol("SexIndex")
val embarkIndexer = new StringIndexer().setInputCol("Embarked").setOutputCol("EmbarkIndex")

// COMMAND ----------

// Convert the numerical index columns into One Hot columns
// The One Hot columns are binary {0,1} values of the categories
val genderEncoder = new OneHotEncoder().setInputCol("SexIndex").setOutputCol("SexVec")
val embarkEncoder = new OneHotEncoder().setInputCol("EmbarkIndex").setOutputCol("EmbarkVec")

// COMMAND ----------

// Create 8 buckets for the fares, turning a continuous feature into a discrete range
val fareSplits = Array(0.0,10.0,20.0,30.0,40.0,60.0,120.0,Double.PositiveInfinity)
val fareBucketize = new Bucketizer().setInputCol("Fare").setOutputCol("FareBucketed").setSplits(fareSplits)

// COMMAND ----------

// Create a vector of the features.  
val assembler = new VectorAssembler().setInputCols(Array("Pclass","SexVec", "Mil", "Doc", "Rev", "Nob", "Mr", "Mrs", "Miss", "Mstr", "TotalFamSize", "Singleton", "SmallFam", "LargeFam", "Child", "Mother","SibSp", "Parch", "EmbarkVec","Age", "Fare", "FareBucketed")).setOutputCol("features")

// COMMAND ----------

// Create the features pipeline and data frame * SAME AS MACHINE LEARNING PIPELINE
// The order is important here, Indexers have to come before the encoders
val FeaturesPipeline = (new Pipeline()
  .setStages(Array(genderIndexer,embarkIndexer,genderEncoder,embarkEncoder, fareBucketize, assembler)))

val trainingFit = FeaturesPipeline.fit(trainCombinedDF)
val trainingFeaturesDF = trainingFit.transform(trainCombinedDF)

val testFeaturesDF = trainingFit.transform(testCombinedDF)

// COMMAND ----------

display(testFeaturesDF)

// COMMAND ----------

// Now that the data has been prepared, let's split the training dataset into a training and validation dataframe
val Array(trainDF, testDF) = trainingFeaturesDF.randomSplit(Array(0.8, 0.2),seed = 12345)

// COMMAND ----------

// Create default param map for XGBoost
def get_param(): mutable.HashMap[String, Any] = {
    val params = new mutable.HashMap[String, Any]()
        params += "eta" -> 0.1
        params += "max_depth" -> 8
        params += "gamma" -> 0.0
        params += "colsample_bylevel" -> 1
        params += "objective" -> "binary:logistic"
        params += "num_class" -> 2
        params += "booster" -> "gbtree"
        params += "num_rounds" -> 20
        params += "nWorkers" -> 3
    return params
}

// COMMAND ----------

// Create an XGBoost Classifier
val xgb = new XGBoostEstimator(get_param().toMap).setLabelCol("Survived").setFeaturesCol("features")

// COMMAND ----------

// XGBoost paramater grid
val xgbParamGrid = (new ParamGridBuilder()
  .addGrid(xgb.round, Array(1000))
  .addGrid(xgb.maxDepth, Array(16))
  .addGrid(xgb.maxBins, Array(2))
  .addGrid(xgb.minChildWeight, Array(0.2))
  .addGrid(xgb.alpha, Array(0.8, 0.9))
  .addGrid(xgb.lambda, Array(0.9, 1.0))
  .addGrid(xgb.subSample, Array(0.6, 0.65, 0.7))
  .addGrid(xgb.eta, Array(0.015))
  .build())

// COMMAND ----------

// Create the XGBoost pipeline
val pipeline = new Pipeline().setStages(Array(xgb))

// COMMAND ----------

// Setup the binary classifier evaluator
val evaluator = (new BinaryClassificationEvaluator()
  .setLabelCol("Survived")
  .setRawPredictionCol("prediction")
  .setMetricName("areaUnderROC"))

// COMMAND ----------

// Create the Cross Validation pipeline, using XGBoost as the estimator, the 
// Binary Classification evaluator, and xgbParamGrid for hyperparameters
val cv = (new CrossValidator()
  .setEstimator(pipeline)
  .setEvaluator(evaluator)
  .setEstimatorParamMaps(xgbParamGrid)
  .setNumFolds(10))

// COMMAND ----------

// Create the model by fitting the training data
val xgbModel = cv.fit(trainDF)

// COMMAND ----------

// Print out a copy of the parameters used by XGBoost
(xgbModel.bestModel.asInstanceOf[PipelineModel]
  .stages(0).asInstanceOf[XGBoostClassificationModel]
  .extractParamMap().toSeq.foreach(println))

// COMMAND ----------

evaluator.evaluate(xgbModel.transform(trainDF))

// COMMAND ----------

// Test the validation data by scoring the model
val results = xgbModel.transform(testDF)
evaluator.evaluate(results)

// COMMAND ----------

// Test data by scoring the model
val scoredDf = xgbModel.transform(testFeaturesDF)

// COMMAND ----------

scoredDf.createGlobalTempView("scoredDFS")

// COMMAND ----------

display(scoredDf)


// COMMAND ----------

scoredDf
      .withColumn("Survived", col("Prediction"))
      .select("PassengerId", "Survived")
      .coalesce(1)
      .write
      .format("csv")
      .option("header", "true")
  
