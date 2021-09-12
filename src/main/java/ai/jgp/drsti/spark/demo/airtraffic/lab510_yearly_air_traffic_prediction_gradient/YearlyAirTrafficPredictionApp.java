package ai.jgp.drsti.spark.demo.airtraffic.lab510_yearly_air_traffic_prediction_gradient;

import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.sum;
import static org.apache.spark.sql.functions.year;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.feature.VectorIndexer;
import org.apache.spark.ml.feature.VectorIndexerModel;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.ml.regression.GBTRegressionModel;
import org.apache.spark.ml.regression.GBTRegressor;
import org.apache.spark.ml.regression.RegressionModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ai.jgp.drsti.spark.DrstiLineChart;
import ai.jgp.drsti.spark.DrstiUtils;
import ai.jgp.drsti.spark.utils.DataframeUtils;

/**
 * 
 * @author jgp
 *
 */
public class YearlyAirTrafficPredictionApp {
  private static Logger log =
      LoggerFactory.getLogger(YearlyAirTrafficPredictionApp.class);

  public static void main(String[] args) {
    YearlyAirTrafficPredictionApp app = new YearlyAirTrafficPredictionApp();
    app.start();
  }

  /**
   * Real work goes here...
   */
  private boolean start() {
    log.debug("-> start()");
    long t0 = System.currentTimeMillis();

    // Creates a session on a local master
    SparkSession spark = SparkSession.builder()
        .appName("CSV to Dataset")
        .master("local[*]")
        .getOrCreate();

    long tc = System.currentTimeMillis();
    log.info("Spark master available in {} ms.", (tc - t0));
    t0 = tc;

    // Reading gold data from Delta
    Dataset<Row> goldDf = spark.read().format("delta")
        .load("./data/tmp/airtrafficmonth")
        .orderBy(col("month"));

    tc = System.currentTimeMillis();
    log.info("Reading gold zone in {} ms.", (tc - t0));
    t0 = tc;

    Dataset<Row> dfYear = goldDf
        .withColumn("year", year(col("month")))
        .groupBy(col("year"))
        .agg(sum("pax").as("pax"),
            sum("internationalPax").as("internationalPax"),
            sum("domesticPax").as("domesticPax"))
        .orderBy(col("year"));

    dfYear = DrstiUtils.setHeader(dfYear, "year", "Year");
    dfYear = DrstiUtils.setHeader(dfYear, "pax", "Passengers");
    dfYear = DrstiUtils.setHeader(dfYear, "internationalPax",
        "International Passengers");
    dfYear =
        DrstiUtils.setHeader(dfYear, "domesticPax", "Domestic Passengers");
    tc = System.currentTimeMillis();
    log.info("Transformation for yearly graph {} ms.", (tc - t0));
    t0 = tc;

    dfYear.show(5, false);
    dfYear.printSchema();

    // // dfYear = indexer.
    // dfYear.show(200, false);
    //
    // // Start a GBTRegressor
    // GBTRegressor gbt = new GBTRegressor()
    // .setLabelCol("pax")
    // .setFeaturesCol("features")
    // .setMaxIter(150)
    // .setLossType("absolute")
    // .setFeatureSubsetStrategy("all");
    //
    // RegressionEvaluator evaluator = new RegressionEvaluator()
    // .setLabelCol("pax")
    // .setPredictionCol("prediction")
    // .setMetricName("rmse");
    //
    // ParamMap[] paramGrid =
    // new ParamGridBuilder().addGrid(gbt.maxDepth(), new int[] { 2, 5 })
    // .addGrid(gbt.maxIter(), new int[] { 10, 100 }).build();
    //
    // CrossValidator cv = new CrossValidator().setEstimator(gbt)
    // .setEvaluator(evaluator).setEstimatorParamMaps(paramGrid);
    //
    // Pipeline pipeline = new Pipeline()
    // .setStages(new PipelineStage[] { assembler, indexer, cv });
    //
    // // Train model
    // GBTRegressionModel model = pipeline.fit(dfYear);
    //
    // // Measures quality index for training data
    // Dataset<Row> predictionsDf = model.transform(dfYear);
    // double rmse = evaluator.evaluate(predictionsDf);
    // if (log.isDebugEnabled()) {
    // log.debug("Root Mean Squared Error (RMSE): {}", rmse);
    // log.debug("Learned regression GBT model:\n{}",
    // model.toDebugString());
    // }

    String[] inputCols = new String[1];
    inputCols[0] = "year";
    VectorAssembler assembler = new VectorAssembler()
        .setInputCols(inputCols)
        .setOutputCol("rawFeatures");
    dfYear = assembler.transform(dfYear);

    VectorIndexerModel featureIndexer = new VectorIndexer()
        .setInputCol("rawFeatures")
        .setOutputCol("indexedFeatures")
        .setMaxCategories(4)
        .fit(dfYear);

    Dataset<Row> trainingData = dfYear.filter(col("year").$less$eq(2015));
    Dataset<Row> testData = dfYear.filter(col("year").$greater(2015));

    // Train a GBT model.
    GBTRegressor gbt = new GBTRegressor()
        .setLabelCol("pax")
        .setFeaturesCol("indexedFeatures")
        .setMaxIter(10);

    // Chain indexer and GBT in a Pipeline.
    Pipeline pipeline = new Pipeline()
        .setStages(new PipelineStage[] { featureIndexer, gbt });

    // Train model. This also runs the indexer.
    PipelineModel model = pipeline.fit(trainingData);

    // Make predictions.
    Dataset<Row> predictions = model.transform(testData);

    // Select example rows to display.
    predictions.show(20);

    RegressionEvaluator evaluator = new RegressionEvaluator()
        .setLabelCol("pax")
        .setPredictionCol("prediction")
        .setMetricName("rmse");
    double rmse = evaluator.evaluate(predictions);
    System.out
        .println("Root Mean Squared Error (RMSE) on test data = " + rmse);

    GBTRegressionModel gbtModel = (GBTRegressionModel) (model.stages()[1]);

    predict(2021, gbtModel);
    predict(2020, gbtModel);
    predict(2019, gbtModel);
    predict(2018, gbtModel);
    predict(2017, gbtModel);

    // Graph
    dfYear.printSchema();
    dfYear = dfYear.drop("features");
    dfYear = dfYear.drop("rawFeatures");
    DrstiLineChart d = new DrstiLineChart(dfYear);
    d.setTitle("US air traffic, in passengers, per year");
    d.setXTitle("Year " + DataframeUtils.min(dfYear, "year") + " - " +
        DataframeUtils.max(dfYear, "year")
        + " - Data cached in Delta Lake");
    d.setYTitle("Passengers (000s)");
    d.render();

    tc = System.currentTimeMillis();
    log.info("Data exported for in {} ms.", (tc - t0));

    spark.stop();
    return true;
  }

  private double predict(double feature, RegressionModel model) {
    double p = model.predict(Vectors.dense(feature));
    log.info("Passengers in year {}: {}", feature, p);
    return p;
  }
}
