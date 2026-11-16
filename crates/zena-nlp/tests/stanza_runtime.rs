//! Stanza 실행기 (src/stanza.rs, stanza-onnx-export format 2 변환본) — Qntology/logis-center `tests/it/stanza_runtime.rs` 에서 이식.
//! 라틴 사전 투표 시험은 raon 사전에 언어 구역이 없어 뺐습니다.
//!
//! - 의존 구문 디코딩: stanza 원본 `chuliu_edmonds_one_root` · `resolve_head_constraint_violations` 출력(tests/fixtures/stanza/decode_cases.json)과 같은 트리
//! - 질의 언어 판정(따옴표 · 문장 속 다른 문자 이름 제외) · 한국어 어간 · MWT 사전 대소문자 규칙

#![cfg(feature = "stanza")]

use serde_json::Value;
use std::collections::HashMap;
use zena_nlp::stanza::{
    chuliu_edmonds_one_root, detect_query_lang, dict_expansion, korean_stem, lang_code, resolve_head_constraint_violations, strip_quoted,
    StanzaLang,
};

fn cases() -> Value {
    serde_json::from_str(include_str!("fixtures/stanza/decode_cases.json")).expect("decode_cases.json")
}

fn matrix(v: &Value) -> Vec<Vec<f64>> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|row| {
            row.as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap_or(f64::NEG_INFINITY))
                .collect()
        })
        .collect()
}

fn ints(v: &Value) -> Vec<usize> {
    v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect()
}

#[test]
fn chuliu_edmonds_matches_stanza() {
    let c = cases();
    for (i, case) in c["cle"].as_array().unwrap().iter().enumerate() {
        let tree = chuliu_edmonds_one_root(&matrix(&case["scores"]));
        assert_eq!(tree, ints(&case["tree"]), "사례 {i}");
        assert_eq!(tree.iter().skip(1).filter(|&&h| h == 0).count(), 1, "루트는 하나");
    }
}

#[test]
fn head_constraints_match_stanza() {
    let c = cases();
    let deprels: Vec<String> = c["deprels"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_str().unwrap().to_string())
        .collect();
    let groups: Vec<Option<&'static str>> = deprels
        .iter()
        .map(|d| match d.as_str() {
            "nsubj" => Some("subj"),
            "obj" => Some("obj"),
            _ => None,
        })
        .collect();
    let mut repaired = 0;
    for (i, case) in c["heads"].as_array().unwrap().iter().enumerate() {
        let scores = matrix(&case["scores"]);
        let labels: Vec<Vec<Vec<f64>>> = case["labels"].as_array().unwrap().iter().map(|m| matrix(m)).collect();
        let plain = chuliu_edmonds_one_root(&scores);
        assert_eq!(plain, ints(&case["plain_tree"]), "사례 {i} 머리 제약 전");
        let (tree, raw) = resolve_head_constraint_violations(&scores, &labels, plain.clone(), &groups, 5);
        assert_eq!(tree, ints(&case["tree"]), "사례 {i} 트리");
        assert_eq!(raw, ints(&case["raw"]), "사례 {i} 관계");
        repaired += (tree != plain) as usize;
    }
    assert!(repaired > 0, "고쳐지는 사례가 하나는 있어야 이 시험이 의미가 있습니다");
}

#[test]
fn query_language_and_codes() {
    assert_eq!(detect_query_lang("5만원 이하 베이지 가디건", "english"), Some("ko"));
    assert_eq!(detect_query_lang("先週キャンセルされた注文", "korean"), Some("ja"));
    assert_eq!(detect_query_lang("货物的保险单", "korean"), Some("zh-hans"));
    assert_eq!(detect_query_lang("貨物の保険証券", "korean"), Some("ja"));
    assert_eq!(detect_query_lang("شهادة المنشأ للبضائع", "korean"), Some("ar"));
    // 그 언어에만 쓰는 글자 · 8개 라틴 언어로 좁힌 판정 (넓게 두면 라틴어 · 슬로바키아어로 판정되어 영어로 떨어졌음)
    assert_eq!(detect_query_lang("zrušené objednávky", "korean"), Some("cs"));
    assert_eq!(detect_query_lang("stornierte Bestellungen für Kunden", "korean"), Some("de"));
    assert_eq!(detect_query_lang("заказы", "korean"), None);
    assert_eq!(lang_code("czech"), Some("cs"));
    assert_eq!(lang_code("chinese"), Some("zh-hans"));
    assert_eq!(lang_code("russian"), None);
}

/// 데이터 값(행사 · 고객 · 쇼핑몰 · 수출자 이름)의 문자 체계가 질의 문장의 언어를 덮지 않는가 (v2 관계형 합성 데이터 질의).
/// 이전에는 영어 문장에 든 「夏休み準備セール」 · 加藤 芽依 · 이예준 때문에 질의가 일본어 · 중국어 · 한국어로 판정되어
/// 그 언어의 사전 · 변환본으로 풀렸습니다 (夏 → 여름 기간 조건 등).
#[test]
fn query_language_ignores_quoted_and_embedded_names() {
    let latin = |q: &str| {
        let l = detect_query_lang(q, "korean");
        assert!(
            matches!(l, Some("en" | "de" | "es" | "fr" | "it" | "pt" | "nl" | "cs")),
            "{q} → {l:?}"
        );
    };
    // 따옴표로 묶은 이름
    assert_eq!(
        strip_quoted("products featured in the '夏休み準備セール' promotion"),
        "products featured in the   promotion"
    );
    assert_eq!(
        strip_quoted("polizze di carico dell'ordine d'acquisto PO-KR-86767"),
        "polizze di carico dell'ordine d'acquisto PO-KR-86767"
    );
    assert_eq!(
        strip_quoted("「夏休み準備セール」"),
        "「夏休み準備セール」",
        "따옴표 밖에 글자가 없으면 원문"
    );
    latin("products featured in the '夏休み準備セール' promotion");
    latin("Artikel der Aktion „夏休み準備セール“");
    latin("produits de l'opération « 夏休み準備セール »");
    assert_eq!(detect_query_lang("“夏休み準備セール”活动的参与商品", "korean"), Some("zh-hans"));
    assert_eq!(
        detect_query_lang("'Bereit für den Sommerurlaub' 기간에 팔린 상품", "korean"),
        Some("ko")
    );
    assert_eq!(
        detect_query_lang("「Summer getaway picks」のイベントページでクーポンを受け取った記録", "korean"),
        Some("ja")
    );
    // 따옴표 없는 이름: 라틴 · 아랍 문장 속 한중일 이름 어절
    latin("products 이예준 wrote a review for");
    latin("shipments delivered to 加藤 芽依 in September");
    latin("bills of lading shipped by 株式会社ひなた工房");
    latin("prodotti recensiti da 이예준");
    assert_eq!(detect_query_lang("المنتجات التي كتب 이예준 مراجعة لها", "korean"), Some("ar"));
    // 한국어 · 일본어 · 중국어 문장 속 이름
    assert_eq!(detect_query_lang("株式会社ひなた工房에서 보낸 선하증권", "korean"), Some("ko"));
    assert_eq!(
        detect_query_lang("こもれび商店 상품 상세 페이지에서 옵션을 바꾼 기록", "korean"),
        Some("ko")
    );
    assert_eq!(detect_query_lang("Felix Wagner 고객이 주문한 상품", "english"), Some("ko"));
    assert_eq!(detect_query_lang("4월에 Fleece-lined zip hoodie 포함된 주문", "korean"), Some("ko"));
    assert_eq!(detect_query_lang("4月にFleece-lined zip hoodieを含む注文", "korean"), Some("ja"));
    assert_eq!(
        detect_query_lang("소래마켓で決済ページまで進んだのに支払わなかった記録", "korean"),
        Some("ja")
    );
    assert_eq!(detect_query_lang("注文書PO-KR-86767の船荷証券", "korean"), Some("ja"));
    assert_eq!(
        detect_query_lang("在こもれび商店商品详情页更改选项的记录", "korean"),
        Some("zh-hans")
    );
    assert_eq!(detect_query_lang("株式会社ひなた工房发运的提单", "korean"), Some("zh-hans"));
    assert_eq!(
        detect_query_lang("在소래마켓进入结算页面却没有付款的记录", "korean"),
        Some("zh-hans")
    );
    // 짧은 로망스어 문장(낱말 셋) 속 두 어절 한자 이름: 문장 낱말이 이름 어절보다 많으면 이름을 빼고 정합니다 (예전엔 중국어로 판정, v2 C8)
    assert_eq!(detect_query_lang("productos que 加藤 芽依 reseñó", "korean"), Some("es"));
    // 가나 없는 한자만: 일본어 신자체(気 · 圧)가 간체보다 많으면 일본어 (앱 언어가 일본어가 아니어도), 간체가 많으면 중국어
    assert_eq!(detect_query_lang("主蒸気圧力", "korean"), Some("ja"));
    assert_eq!(detect_query_lang("货物的保险单", "korean"), Some("zh-hans"));
    // 숫자가 든 번호의 로마자는 문장 언어가 아님 (아랍어 문장 + 서류 번호)
    assert_eq!(detect_query_lang("مستندات BL-DE-260911", "korean"), Some("ar"));
}

#[test]
fn korean_stems_and_mwt_casing() {
    assert_eq!(korean_stem("가디건+을", "ncn+jco"), "가디건");
    assert_eq!(korean_stem("취소+되+ㄴ", "ncpa+xsv+etm"), "취소");
    assert_eq!(korean_stem("5만+원", "nnc+nbu"), "5만원");
    assert_eq!(korean_stem("찾+아+주+어", "pvg+ecx+px+ef"), "찾");
    let d: HashMap<String, String> = [("del".to_string(), "de el".to_string()), ("zum".to_string(), "zu dem".to_string())]
        .into_iter()
        .collect();
    assert_eq!(dict_expansion(&d, "del").as_deref(), Some("de el"));
    assert_eq!(dict_expansion(&d, "DEL").as_deref(), Some("DE EL"));
    assert_eq!(dict_expansion(&d, "Zum").as_deref(), Some("Zu dem"));
    assert_eq!(dict_expansion(&d, "zUm"), None);
}
