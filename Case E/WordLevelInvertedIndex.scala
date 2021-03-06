import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._

object WordLevelInvertedIndex {
  def main(args: Array[String]) = {
    val sc = SparkContext.getOrCreate()
    val fileName = "FileStore/tables/main.txt" // Ran on databricks.
    val result = sc.textFile(fileName)
                   .map{ line => // Split the lines on :, such that we end up with the doc_id and the text.
                      val array = line.split(":", 2)
                      (array(0).toInt, array(1))
                    } 
                  .flatMap { // flatMap split on one or more non-word characters, RDD is now a column of words, and a column of doc_id's that contain those words. Also calculate location inside the document.
                    case (doc_id, text) =>
                      val text2 = text.toLowerCase() // Do some minor preprocessing, real life circumstances would warrant more preprocessing such as stemming, removal of punctuation, stopword removal
                      text2.split("""\W+""") map {
                        word => (word, doc_id, text2.indexOfSlice(word))
                      }
                  } 
                  .map { // Add initial count for each of the words
                    case (word, doc_id, pos) => ((word, doc_id, pos), 1)
                  } 
                  .reduceByKey { // Combine the counts and group by them
                    case (n1, n2) => n1 + n2
                  } 
                  .map { // Use the word as the key
                    case ((word, doc_id, pos), n) => (word, (doc_id,pos,n))
                  } 
                  .groupBy { // Group by word such that we can connect all the docs that contain the word
                    case (word, (doc_id,pos,n)) => word
                  } 
                  .map { // Finally all the grouped sequences of doc_id, pos, n are added to maps.
                    case (word, seq) => 
                      val seq2 = seq map {
                        case (_, (doc_id, pos, n)) => HashMap[String,Integer](("doc_id", doc_id), ("pos", pos), ("count", n))
                      }
                    (word, seq2.toVector) // Convert sequence of maps to vector such that it can be output
                  }
    result.coalesce(1, true).saveAsTextFile("FileStore/tables/result12356.txt")
    val df = spark.createDataFrame(result)
    val finaldf = df.toDF("word", "doc_id, pos, count")
    finaldf.show()
  }
}

WordLevelInvertedIndex.main(Array())