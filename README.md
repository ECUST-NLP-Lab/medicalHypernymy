# Medical Hypernymy Detection Dataset

This repository provides two corpora which contain hypernymy pairs of medical terms, both in English and Chinese. The corpora also contain negative instances, and have been splited into training sets, test sets and validation sets.

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

## English Corpus
We extract terms which are labeled as "clinical finding" in SNOMED CT, and their children to construct positive hypernymy instances. We also take hyponymy, synonymy and unrelated pairs as negative instances. 

## Chinese Corpus
We select six Chinese healthcare websites, and extract hypernymy and synonymy relations between symptoms from semi-structured and unstructured data on the detail pages. We set hypernymy symptom pairs as positive instances, hyponymy, synonymy and unrelated symptom pairs as negative instances.

### Six Chinese Healthcare Websites
<table class="tg">
  <tr>
    <th>Website</th>
    <th>URL</th>
  </tr>
  <tr>
    <th>XYWY</th>
    <th><a href="http://www.xywy.com/">http://www.xywy.com/</a></th>
  </tr>
  <tr>
    <td>120ask</td>
    <td><a href="http://www.120ask.com/">http://www.120ask.com/</a></td>
  </tr>
  <tr>
    <td>39Health<br></td>
    <td><a href="http://www.39.net/">http://www.39.net/</a></td>
  </tr>
  <tr>
    <td>99Health</td>
    <td><a href="http://www.99.com.cn/">http://www.99.com.cn/</a></td>
  </tr>
  <tr>
    <td>Familydoctor</td>
    <td><a href="http://www.familydoctor.com.cn/">http://www.familydoctor.com.cn/</a></td>
  </tr>
  <tr>
    <td>Fh21</td>
    <td><a href="http://www.fh21.com.cn/">http://www.fh21.com.cn/</a></td>
  </tr>
</table>
