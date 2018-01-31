# medicalHypernymy

This repository provides two corpora which contain hypernymy pairs of medical terms, both in English and Chinese. The corpora also contain negative instances, and have been splited into training sets, test sets and validation sets.
<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;}
.tg th{font-family:Arial, sans-serif;font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;}
.tg .tg-bzci{font-size:20px;text-align:center;vertical-align:top}
.tg .tg-4kyz{font-size:20px;text-align:center}
</style>
<table class="tg">
  <tr>
    <th class="tg-bzci" colspan="2">Dataset</th>
    <th class="tg-bzci">  Positive  </th>
    <th class="tg-bzci">   Negative   </th>
    <th class="tg-bzci">All</th>
  </tr>
  <tr>
    <td class="tg-4kyz" rowspan="3">     English     </td>
    <td class="tg-4kyz">     Train     </td>
    <td class="tg-4kyz">     27,872   </td>
    <td class="tg-4kyz">27,872</td>
    <td class="tg-bzci">     55,744     </td>
  </tr>
  <tr>
    <td class="tg-4kyz"> Test </td>
    <td class="tg-4kyz">9,954</td>
    <td class="tg-4kyz">9,954</td>
    <td class="tg-bzci">19,908</td>
  </tr>
  <tr>
    <td class="tg-4kyz">Val</td>
    <td class="tg-4kyz">1,991</td>
    <td class="tg-4kyz">1,991</td>
    <td class="tg-bzci">3,982</td>
  </tr>
  <tr>
    <td class="tg-4kyz" rowspan="3">Chinese</td>
    <td class="tg-4kyz">Train</td>
    <td class="tg-4kyz">8,960</td>
    <td class="tg-4kyz">8,960</td>
    <td class="tg-bzci">17,920</td>
  </tr>
  <tr>
    <td class="tg-4kyz">Test</td>
    <td class="tg-4kyz">3,200</td>
    <td class="tg-4kyz">3,200</td>
    <td class="tg-bzci">6,400</td>
  </tr>
  <tr>
    <td class="tg-bzci">Val</td>
    <td class="tg-bzci">640</td>
    <td class="tg-bzci">640</td>
    <td class="tg-bzci">1,280</td>
  </tr>
</table>

##English Corpus
We extract terms which are labeled as "clinical finding" in SNOMED CT, and their children to construct positive hypernymy instances. We also take hyponymy, synonymy and unrelated pairs as negative instances. 

##Chinese Corpus
We select six Chinese healthcare websites, and extract hypernymy and synonymy relations between symptoms from semi-structured and unstructured data on the detail pages. We set hypernymy symptom pairs as positive instances, hyponymy, synonymy and unrelated symptom pairs as negative instances.

Six Chinese Healthcare Websites
<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;}
.tg th{font-family:Arial, sans-serif;font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;}
.tg .tg-s6z2{text-align:center}
</style>
<table class="tg">
  <tr>
    <th class="tg-s6z2">XYWY</th>
	<a href="http://www.xywy.com/">
    <th class="tg-s6z2">http://www.xywy.com/</th>
	</a>
  </tr>
  <tr>
    <td class="tg-s6z2">120ask</td>
	<a href="http://www.120ask.com/">
    <td class="tg-s6z2">http://www.120ask.com/</td>
	</a>
  </tr>
  <tr>
    <td class="tg-s6z2">39Health<br></td>
	<a href="http://www.39.net/">
    <td class="tg-s6z2">http://www.39.net/</td>
	</a>
  </tr>
  <tr>
    <td class="tg-s6z2">99Health</td>
	<a href="http://www.99.com.cn/">
    <td class="tg-s6z2">http://www.99.com.cn/</td>
	</a>
  </tr>
  <tr>
    <td class="tg-s6z2">      Familydoctor      </td>
	<a href="http://www.familydoctor.com.cn/">
    <td class="tg-s6z2">         http://www.familydoctor.com.cn/         </td>
	</a>
  </tr>
  <tr>
    <td class="tg-s6z2">Fh21</td>
	<a href="http://www.fh21.com.cn/">
    <td class="tg-s6z2">http://www.fh21.com.cn/</td>
	</a>
  </tr>
</table>
