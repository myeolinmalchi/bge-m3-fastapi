from bs4 import BeautifulSoup, Tag
from bs4.element import NavigableString, PageElement


def process_string(soup: BeautifulSoup, element: PageElement) -> int:
    if not element.parent and element != soup:
        return 0

    if len(element.text) == 0 or all(c == "" for c in element.text):
        element.extract()
        return 1

    return 0


def process_tag(soup: BeautifulSoup, element: Tag) -> int:

    if not element.parent and element != soup:
        return 0

    affected = 0

    children = list(element.children)
    only_string = all(isinstance(child, NavigableString) for child in children)

    if only_string and len(children) > 0:
        inner_texts = [child.text for child in children]
        combined_text = ''.join(inner_texts)

        element.clear()
        element.append(NavigableString(combined_text))

        affected += 1

    match element:
        case Tag(name="a", attrs={"href": str(href)}):
            if not href.startswith("#"):
                return affected
            target = soup.select_one(href)
            if not target:
                return affected
            target.extract()
            element.replace_with(target)
            return affected + 1
        case Tag(
            name="span" | "p" | "u" | "b" | "strong",
            parent=Tag(name="span" | "p" | "td" | "li" | "td" | "th" | "b")
        ):
            if not only_string:
                return affected
            inner_text = element.get_text(strip=True)
            element.replace_with(NavigableString(inner_text))
            return affected + 1
        case _ if len(children) == 0 or element.text == "":
            element.extract()
            return affected + 1
        case _:
            return affected


def clean_html(html: str | BeautifulSoup) -> BeautifulSoup:
    match html:
        case str():
            html = html.replace("<br/>", " ")
            html = html.replace("<br />", " ")
            html = html.replace("<br>", " ")
            soup = BeautifulSoup(html, "html.parser")
        case BeautifulSoup():
            soup = html

    while True:
        affected = sum([
            process_tag(soup, child)
            if isinstance(child, Tag) else process_string(soup, child)
            for child in list(soup.descendants)
        ])

        if affected == 0:
            break

    return soup
