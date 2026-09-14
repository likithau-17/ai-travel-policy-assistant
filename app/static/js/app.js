const form = document.getElementById("question-form");
const questionInput = document.getElementById("question");
const chat = document.getElementById("chat");

function addMessage(label, text) {
    const paragraph = document.createElement("p");

    const strong = document.createElement("strong");
    strong.textContent = `${label}:`;

    paragraph.appendChild(strong);
    paragraph.appendChild(document.createTextNode(` ${text}`));

    chat.appendChild(paragraph);
}

function addListItem(label, value) {
    const item = document.createElement("li");

    const strong = document.createElement("strong");
    strong.textContent = `${label}:`;

    item.appendChild(strong);
    item.appendChild(document.createTextNode(` ${value}`));

    return item;
}

function formatResult(result) {
    if (result.answer) {
        addMessage("Assistant", result.answer);

        if (result.sources && result.sources.length > 0) {
            const heading = document.createElement("p");
            const strong = document.createElement("strong");
            strong.textContent = "Sources:";
            heading.appendChild(strong);
            chat.appendChild(heading);

            const list = document.createElement("ul");

            result.sources.forEach((source) => {
                const item = document.createElement("li");
                item.textContent = `${source.source} — ${source.section}`;
                list.appendChild(item);
            });

            chat.appendChild(list);
        }

        return;
    }

    if (result.employee_id && result.eligibility_status) {
        addMessage("Assistant", "");

        const list = document.createElement("ul");

        list.appendChild(addListItem("Employee", result.employee_id));
        list.appendChild(addListItem("Country", result.country));
        list.appendChild(addListItem("Employment Type", result.employment_type));
        list.appendChild(addListItem("Eligibility", result.eligibility_status));
        list.appendChild(addListItem("Manager Approval", result.manager_approval));

        chat.appendChild(list);
        return;
    }

    if (result.status) {
        addMessage("Assistant", "");

        const list = document.createElement("ul");

        list.appendChild(addListItem("Status", result.status));
        list.appendChild(
            addListItem(
                "Reason",
                result.reason || result.message || "No additional information."
            )
        );

        if (result.standard_limit !== undefined) {
            list.appendChild(
                addListItem("Standard Limit", result.standard_limit)
            );
        }

        if (result.trip_amount !== undefined) {
            list.appendChild(
                addListItem("Trip Amount", result.trip_amount)
            );
        }

        if (result.reimbursable_amount !== undefined) {
            list.appendChild(
                addListItem(
                    "Reimbursable Amount",
                    result.reimbursable_amount
                )
            );
        }

        chat.appendChild(list);
        return;
    }

    addMessage("Assistant", JSON.stringify(result));
}

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const question = questionInput.value.trim();

    if (!question) {
        return;
    }

    addMessage("You", question);

    questionInput.value = "";

    try {
        const response = await fetch("/ask", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                question: question
            })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Request failed.");
        }

        formatResult(data.result);

    } catch (error) {
        addMessage("Error", error.message);
    }
});