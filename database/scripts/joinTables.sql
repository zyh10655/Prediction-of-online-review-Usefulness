-- extract all relevant data for analysis
-- EXCLUDES: closed businesses, zero usefulness reviews

SELECT
    r.review_id AS r_id,
    r.stars AS r_stars,
    r.date AS r_date,
    r.text AS r_text,
    r.useful AS r_useful,
    r.funny AS r_funny,
    r.cool AS r_cool,
    b.stars AS b_stars,
    b.review_count AS b_review_count,
    b.is_open AS b_is_open,
    u.review_count AS u_review_count,
    u.yelping_since AS u_yelping_since,
    u.friends AS u_friends
FROM review AS r
LEFT JOIN business AS b
ON r.business_id=b.business_id
LEFT JOIN user AS u
ON r.user_id=u.user_id
WHERE b.is_open<>0 AND r.useful<>0;
