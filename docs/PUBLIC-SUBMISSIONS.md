# Public submission results

These board scores and hashes are extracted from the private submission diary.
Private storage routes, receipts, and worker notes are withheld. No value is refitted. The best scored entry is v73, RMSE 223.3071.
The release target `zabd` (candidate SHA-256 `e7b04ffdd02a27d48dfcbf12fd3f682f7adf3fbef711dac5e34125296804a636`) was uploaded as v62 and scored 225.9106.
Release target `v64` (candidate SHA-256 `9cd421ddc503ef528d7211505436549e44498d015594d37a9080df562050571b`) is zabd with the T11b-L stage; it was uploaded as v64 and scored 225.7082.
The final entry, release target `v73` (candidate SHA-256 `70236754db85e5f23fac281ebbc9c14a3a4fe33c4e6f34e7c95262eebe1659d8`), is v64 with five fixed-scope steps, a level-2 correction on clock rows and a sequence-member column on ordinary rows; it was uploaded as v73 and scored 223.3071.

Uploads v26-v29, v31 and v33 were score probes. They are history.
The current rule forbids that use of scores: no released weight, residual, mask or rate
comes from those readings. V30 was not scored. V25 was a failed model candidate.
V38 and v39 were information entries for the held-month versus full-year fit
comparison. V42 to v46 checked the alignment of the two-fold local validation
with the board; v45 was the expected-worse member of an alignment pair and v46 a
null check. V48 to v62 were steps of three stacking ladders, one candidate per step,
each with a forecast frozen before the upload. V63 to v67 were five whole entries,
each read against v62 with a forecast frozen before the upload; v64 scored best.
V68 to v75 were steps on top of v64, built from the v64 release export and the official files,
each with a forecast written before the upload; v73 scored best.
V76 was one board check of target `final` (V): the January and July rows of V (344,841 rows, in
submitting.parquet order), with a forecast written before the upload. The final file (release target
`final-v-dh-r9-ld`) is V15A-R9-LD with the DH step: V with 15 fits of the s7 member, the s7 input of L2C, the
12-fit MLP values on every member-scope row of the assembly and ORD-S on the ordinary rows of long-taxi
airport-days (write-up section 4.11). On these rows it differs from V on 320,292 rows: 320,182 through the four
changes (1,447 of them EDDF January rows that the DH step also moves) and 110 EDDF January rows through the DH
step alone.
No released parameter comes from any of these scores.

| N | submitted file | SHA-256 | board RMSE |
|---|---|---|---|
| 1 | youthful-giraffe_v1.parquet | 48472e441288f0dfe33d35d5a18d31df786b1c4abe7ee06f9ef171ace2f0a6f0 | 265.7612 |
| 2 | youthful-giraffe_v2.parquet | 5d6a99610907bc432e2e46fb9ea11fc8103feb8aa9efdd3f7c86f1188866a4cb | 265.9631 |
| 3 | youthful-giraffe_v3.parquet | 94f18e47a73265a6db66811ededfc3c935fb2a37fac04bf0cd34d45af708a2a1 | 264.3861 |
| 4 | youthful-giraffe_v4.parquet | 08d4c6c019763fbb5a02f363e04f7c85e2cb129612e81a0f78ee5552c655d53f | 275.2197 |
| 5 | youthful-giraffe_v5.parquet | af179065c7d7052daa4eda09392576f4d64072c27d5c2923a7ee8bcbdaddeb5f | 266.2918 |
| 6 | youthful-giraffe_v6.parquet | d55b1de9ce8fe7577c8c566c6efc57301d18484242aaa06ce808a111d38dabb0 | 263.6922 |
| 7 | youthful-giraffe_v7.parquet | 67b70ee379e62a2a3bc9d91c9d771112baf3233efa3f74f4f95adbd7a3c03ef7 | 263.4623 |
| 8 | youthful-giraffe_v8.parquet | a3a139dbe8255d5dfc17aa69cfd4c65a15df26c2e76feb9eeaf4dbad03ed6ba5 | not scored |
| 9 | youthful-giraffe_v9.parquet | a3a139dbe8255d5dfc17aa69cfd4c65a15df26c2e76feb9eeaf4dbad03ed6ba5 | 263.0827 |
| 10 | youthful-giraffe_v10.parquet | ad962763236e9b40ace45a5dac12945568bbce441ed0547ff8e814d248c72c78 | 263.0464 |
| 11 | youthful-giraffe_v11.parquet | 42ca5ee6410938d56c8830e9ec3980d806633c3b2ff15963c0ffe2fe97c39f8c | 263.0298 |
| 12 | youthful-giraffe_v12.parquet | 48dd572d61ff2403f13614ce04bdc3a8fd0dbdb5175bef7eb143650ea520ec52 | 262.7418 |
| 13 | youthful-giraffe_v13.parquet | 79a301627e918140933bb4c1dcf9a9249a63bd81669f1793e252f821aa1dee51 | 248.4818 |
| 14 | youthful-giraffe_v14.parquet | 519ca89f381056afb453ca62d64df622fcdbbdb14d8607cb0307c2be077676b2 | 248.2904 |
| 15 | youthful-giraffe_v15.parquet | eabd824d925f15223db452557fe113e286ca8ce49302e8a97e79f060c0360ea1 | 246.8263 |
| 16 | youthful-giraffe_v16.parquet | 6f8368da9b8da818a726c7cc6cf20de6ff7777196f1e78d41015a8cda441fdf7 | 246.7053 |
| 17 | youthful-giraffe_v17.parquet | 4de323fc4fd9d4865b201b0dd0da1ddcbf7de9e11f1d9bc8af07d293585001ea | 246.6571 |
| 18 | youthful-giraffe_v18.parquet | 4d75706bfcc13d88f7556265cd008b3d3a312555955dca8eb528e4db5325fee8 | 246.3605 |
| 19 | youthful-giraffe_v19.parquet | d03d132a5ff5d3fc70a72b9f60a1b418a169e911ac32ef08b14378695932aa14 | 245.7752 |
| 20 | youthful-giraffe_v20.parquet | 8fd4b34166b61756a20746c77f97fc1d06d06af28ee959ef7deac1de0a365d78 | 245.2901 |
| 21 | youthful-giraffe_v21.parquet | 7b0557a497f8c09e4f56a11352783b9dda37fe1d0579345d4303e4f74d6e7aca | 245.094 |
| 22 | youthful-giraffe_v22.parquet | f11d34f88b62a43ec2a0cdbeec967a9ee4b7d92e50a5508aa4d30e5a78ea5e1c | 245.0207 |
| 23 | youthful-giraffe_v23.parquet | fd420d855fe8217a3915cce42fa7dc7794d2c19d7cb26dbf90ab456650f28220 | 244.1236 |
| 24 | youthful-giraffe_v24.parquet | 984884e1f8d2b5403f7e791659fd9902eacac7206484fb14ed3116004df9dc36 | 242.8064 |
| 25 | youthful-giraffe_v25.parquet | f5015951639ce8863ca8cf1c8591abfa0de6b8b9bcdf885b267a32fe32185624 | 244.0549 |
| 26 | youthful-giraffe_v26.parquet | 7fae2d4a05f919194c95ad6f629d71ffabd1c3238f9542bd4d5cae9abe511a51 | 243.1117 |
| 27 | youthful-giraffe_v27.parquet | ea79ca4d5e01a885c27182f4700f97a0cd4b8233b036997ee58cf835ddadb577 | 243.5443 |
| 28 | youthful-giraffe_v28.parquet | 5c7e39758d932cfdcd877ac4e880c5d8d9c1c67ea40d3d3358771ba34d340559 | 242.9651 |
| 29 | youthful-giraffe_v29.parquet | 24ba73bbc09ab6523ab609d35fb1e2cd56f07748253385a950911400bd13a31d | 247.4948 |
| 30 | youthful-giraffe_v30.parquet | 37122906a6eae848a4c6401b5cd8090ff39be187ddeee41732b69f5315ba7607 | not scored |
| 31 | youthful-giraffe_v31.parquet | 37122906a6eae848a4c6401b5cd8090ff39be187ddeee41732b69f5315ba7607 | 243.7659 |
| 32 | youthful-giraffe_v32.parquet | f09f609429c81fd888e19e75e16f51b57978d55cc2639d6161c894da0baaa0de | 242.7480 |
| 33 | youthful-giraffe_v33.parquet | c9dbe8c4fd11381ebec6b80ed38c0a4bfbaa0c1f35e6405687b69f1cfa295611 | 245.1436 |
| 34 | youthful-giraffe_v34.parquet | 98bf036199b9708e2f0dd9eb8d88f41be16bb56cbff1319b9ea02850112d3125 | 243.5329 |
| 35 | youthful-giraffe_v35.parquet | 4ece6a4b3616a57cc1a54d2fa63657b590226ff052dafc2f791821566b470e35 | 241.7342 |
| 36 | youthful-giraffe_v36.parquet | 3ae49f67d62a832c6da7a25ac7538dc62273a9c5f2f7815778f451a24ab5c32f | 240.0047 |
| 37 | youthful-giraffe_v37.parquet | 2d847eb965ef7b0e58e6655569c2e9546a366b58cad33e7dbfe9c1010097699b | 239.9380 |
| 38 | youthful-giraffe_v38.parquet | 2947f10d5cc24a5e664271ddd23cfd915b2cb002ec47ce307eb193a3ac9bdc14 | 245.4387 |
| 39 | youthful-giraffe_v39.parquet | 8ea697fb839e19f552d505119fd7369ea709fc4871aef85ea3ce7bfb8d4380f1 | 242.8854 |
| 40 | youthful-giraffe_v40.parquet | c29c9dc434e2b2e0b910a382d8fd83a9dd6de3e05cf865203ba1a53383623e45 | 238.9597 |
| 41 | youthful-giraffe_v41.parquet | 10d2f9337911d220e4768a68ed0103a8e09305fde7e757aba46f9b290f8fc434 | 239.5943 |
| 42 | youthful-giraffe_v42.parquet | cf8d291647f227b8107a063b84376cd523f2fadbd14260190fc32edd1235994b | 243.4718 |
| 43 | youthful-giraffe_v43.parquet | 7b29c560ab633254cb0879999aca959de983b6624f3f514f4c73840600a82db0 | 240.0861 |
| 44 | youthful-giraffe_v44.parquet | ccfbab1f09192105f74c485956468f09dd0546c7ca526a5db44926d1c31a5771 | 239.9574 |
| 45 | youthful-giraffe_v45.parquet | a1a7a2f59e1521c74e88025f5aae81bc828179e67196ba6c79021ab51c766e17 | 241.1860 |
| 46 | youthful-giraffe_v46.parquet | 77219c20b78a1ff3658f9fecce13350c79899baa4f19fe293382eaeaaf34730d | 240.1240 |
| 47 | youthful-giraffe_v47.parquet | 3d58e932c9c3c02f43e97ed6bff8deeb1ea9268c04d4fe20ed76626aa41d91a5 | 237.7071 |
| 48 | youthful-giraffe_v48.parquet | 08e2fa18cfc044a605ba8deb1fcae8f28ad75242c49c917346b508f87bee4feb | 235.1964 |
| 49 | youthful-giraffe_v49.parquet | e9771b940f425ed9bd08bab684c3e1f710736934b5ae316aa8773434461b44b0 | 232.9026 |
| 50 | youthful-giraffe_v50.parquet | 4c0d4dc131596d173fe08d2c68c1a55fd3214dbea7948173c6d89906f0aa93dd | 229.7933 |
| 51 | youthful-giraffe_v51.parquet | 46ee2f909cf8db81eee1ff7af06d58fda2fdf9df32128ccfb819f0491b3e9a70 | 228.8603 |
| 52 | youthful-giraffe_v52.parquet | 89f3c83cd1d689f1cf3a193e3c17a0e7e64119ddc1c76b48e7631fba33d1f381 | 227.7205 |
| 53 | youthful-giraffe_v53.parquet | fec00f357276aeb07d33a19955e5c639c544f04547d5e7869ddac43d4f9c4a50 | 227.4400 |
| 54 | youthful-giraffe_v54.parquet | 58cde60ade0baaba638fb4950d5c3c7dde613463595f7dc91e43759138797ce8 | 227.2506 |
| 55 | youthful-giraffe_v55.parquet | f7c4904d3193ce5072820f944f04aca8449e7ebd196c696c49f627f07f2c16fb | 227.2199 |
| 56 | youthful-giraffe_v56.parquet | 3c7e0c5ffefe97c5b66df7dc9222c2d606225f2fe3e30c46a854d5d68eda4e20 | 227.0332 |
| 57 | youthful-giraffe_v57.parquet | 18ff858f107bde936040167b91689603a05a4807d106dd8f5c2e227a4c59330a | 226.9242 |
| 58 | youthful-giraffe_v58.parquet | 22b71023e6729b117c2508d6d7146e8873a664632da49f164f1f42e12a1be54a | 226.3211 |
| 59 | youthful-giraffe_v59.parquet | e6dfedd5f7ae33081decd1f4c7314480f27d57ec9de1ab4ffc417e0fd22c3262 | 225.8463 |
| 60 | youthful-giraffe_v60.parquet | 2dd4bf0e8ff25c6b08e0745c9994851484b29632b863d15e9fd569c60536d576 | 226.0755 |
| 61 | youthful-giraffe_v61.parquet | 9436b85b0c5b2c83dd4b6b1be16076166337e7c274a3d4e6a59481843fd9cf9c | 226.0149 |
| 62 | youthful-giraffe_v62.parquet | e7b04ffdd02a27d48dfcbf12fd3f682f7adf3fbef711dac5e34125296804a636 | 225.9106 |
| 63 | youthful-giraffe_v63.parquet | 1b0b38a11a231539b8d6817d4e96f593d17496c36ae54ae31119bf34a206194f | 225.9127 |
| 64 | youthful-giraffe_v64.parquet | 9cd421ddc503ef528d7211505436549e44498d015594d37a9080df562050571b | 225.7082 |
| 65 | youthful-giraffe_v65.parquet | 91414c639486c2a9385d18ffa361dce792b4ca183fc4279f23c004b642013e8c | 225.8717 |
| 66 | youthful-giraffe_v66.parquet | 71db15c9f83f291f7865a400fc3f8af127a88650ca460afb0ebaf301e6bb5960 | 225.7757 |
| 67 | youthful-giraffe_v67.parquet | aa75b0292b135d67654d30fc208f9977de00b5e962af2224550fd32e31ecc3d1 | 225.7162 |
| 68 | youthful-giraffe_v68.parquet | 6385025469ad60ca87b27dda1fddc2c09344dd85c2df92c8578e6e6107475ab3 | 224.243 |
| 69 | youthful-giraffe_v69.parquet | 3922dc2ac03dc86bd3e951ad5c931548e9936cf9c86bafccb4885bde5764bbe6 | 223.8572 |
| 70 | youthful-giraffe_v70.parquet | 5600b874c339d42d055570ffd38895b4853b017a92b550030b2914445c79f1d7 | 223.743 |
| 71 | youthful-giraffe_v71.parquet | 0014a5e04842061dca8e860d3f3a4aa5176dd6978a650c50e4214f52e8af75af | 223.6029 |
| 72 | youthful-giraffe_v72.parquet | 712e200bd0a2960936d7b3594437e7d44e6290188b8cc52f67517ef0ea2129bf | 223.5452 |
| 73 | youthful-giraffe_v73.parquet | 70236754db85e5f23fac281ebbc9c14a3a4fe33c4e6f34e7c95262eebe1659d8 | 223.3071 |
| 74 | youthful-giraffe_v74.parquet | feb3a157890fdf4ac3a26ba650bfa3e769e3488c852ece05da391262d42a9eaa | 224.2436 |
| 75 | youthful-giraffe_v75.parquet | caf22c218d79ec7005744afafb378656c2018cfdd141140696d65e7b515ea7e6 | 223.5121 |
| 76 | youthful-giraffe_v76.parquet | 7c398d26dcdd6339f30f986c837d58fd2da3b0e607ff5104e68eb7983905baa9 | 223.8069 |
